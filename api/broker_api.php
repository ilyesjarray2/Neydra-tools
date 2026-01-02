<?php
/**
 * Broker API Aggregator
 * Connects to multiple broker APIs for real-time XAU/USD data
 * NEYDRA Platform - Enterprise Grade
 * Founder & CEO: Ilyes Jarray
 * © 2025 - All Rights Reserved
 */

// ===== ERROR HANDLING & LOGGING =====
error_reporting(E_ALL);
ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', __DIR__ . '/logs/broker_api.log');

// Create logs directory if not exists
if (!is_dir(__DIR__ . '/logs')) {
    mkdir(__DIR__ . '/logs', 0755, true);
}

date_default_timezone_set('UTC');

// ===== CONFIGURATION =====
class Config {
    // OANDA Configuration
    public static $OANDA = [
        'enabled' => getenv('OANDA_ENABLED') ?: true,
        'endpoint' => 'https://api-fxpractice.oanda.com/v3',
        'api_key' => getenv('OANDA_API_KEY') ?: 'YOUR_OANDA_API_KEY',
        'account_id' => getenv('OANDA_ACCOUNT_ID') ?: 'YOUR_ACCOUNT_ID',
        'instruments' => 'XAU_USD',
        'timeout' => 10
    ];

    // Alpha Vantage Configuration
    public static $ALPHA_VANTAGE = [
        'enabled' => getenv('ALPHAVANTAGE_ENABLED') ?: true,
        'endpoint' => 'https://www.alphavantage.co/query',
        'api_key' => getenv('ALPHAVANTAGE_API_KEY') ?: 'XX7L2K09IBKCD5SH',
        'timeout' => 10
    ];

    // Finnhub Configuration
    public static $FINNHUB = [
        'enabled' => getenv('FINNHUB_ENABLED') ?: true,
        'endpoint' => 'https://finnhub.io/api/v1',
        'api_key' => getenv('FINNHUB_API_KEY') ?: 'd5c18i9r01qsbmght95gd5c18i9r01qsbmght960',
        'timeout' => 10
    ];

    // IEX Cloud Configuration
    public static $IEX_CLOUD = [
        'enabled' => getenv('IEX_ENABLED') ?: true,
        'endpoint' => 'https://cloud.iexapis.com/stable',
        'api_key' => getenv('IEX_API_KEY') ?: 'YOUR_IEX_KEY',
        'timeout' => 10
    ];

    // Cache Configuration
    public static $CACHE = [
        'enabled' => true,
        'ttl' => 60, // seconds
        'dir' => __DIR__ . '/cache'
    ];

    // Rate Limiting
    public static $RATE_LIMIT = [
        'enabled' => true,
        'requests_per_minute' => 100,
        'cache_key' => 'rate_limit'
    ];
}

// ===== LOGGING CLASS =====
class Logger {
    private static $log_file = __DIR__ . '/logs/broker_api.log';

    public static function log($level, $message, $context = []) {
        $timestamp = date('Y-m-d H: i:s');
        $context_str = ! empty($context) ? json_encode($context) : '';
        $log_message = "[{$timestamp}] [{$level}] {$message} {$context_str}\n";

        error_log($log_message, 3, self::$log_file);
        
        // Also output to stdout in development
        if (PHP_SAPI === 'cli') {
            echo $log_message;
        }
    }

    public static function info($message, $context = []) {
        self::log('INFO', $message, $context);
    }

    public static function warning($message, $context = []) {
        self::log('WARNING', $message, $context);
    }

    public static function error($message, $context = []) {
        self::log('ERROR', $message, $context);
    }

    public static function debug($message, $context = []) {
        self::log('DEBUG', $message, $context);
    }
}

// ===== CACHE CLASS =====
class Cache {
    public static function get($key) {
        if (! Config::$CACHE['enabled']) {
            return null;
        }

        $cache_dir = Config::$CACHE['dir'];
        if (!is_dir($cache_dir)) {
            mkdir($cache_dir, 0755, true);
        }

        $file = $cache_dir . '/' .  md5($key) . '.cache';

        if (! file_exists($file)) {
            return null;
        }

        $data = unserialize(file_get_contents($file));
        
        // Check if cache expired
        if ($data['expires'] < time()) {
            unlink($file);
            return null;
        }

        return $data['value'];
    }

    public static function set($key, $value, $ttl = null) {
        if (!Config:: $CACHE['enabled']) {
            return;
        }

        $ttl = $ttl ?: Config::$CACHE['ttl'];
        $cache_dir = Config::$CACHE['dir'];
        
        if (!is_dir($cache_dir)) {
            mkdir($cache_dir, 0755, true);
        }

        $file = $cache_dir . '/' . md5($key) . '.cache';
        $data = [
            'value' => $value,
            'expires' => time() + $ttl
        ];

        file_put_contents($file, serialize($data));
    }

    public static function clear($key = null) {
        $cache_dir = Config::$CACHE['dir'];

        if (! is_dir($cache_dir)) {
            return;
        }

        if ($key) {
            $file = $cache_dir . '/' . md5($key) . '.cache';
            if (file_exists($file)) {
                unlink($file);
            }
        } else {
            $files = glob($cache_dir . '/*.cache');
            foreach ($files as $file) {
                unlink($file);
            }
        }
    }
}

// ===== RATE LIMITER CLASS =====
class RateLimiter {
    public static function isAllowed($identifier = 'global') {
        if (!Config::$RATE_LIMIT['enabled']) {
            return true;
        }

        $cache_key = Config::$RATE_LIMIT['cache_key'] . ':' . $identifier;
        $current = Cache::get($cache_key) ?: 0;
        $limit = Config::$RATE_LIMIT['requests_per_minute'];

        if ($current >= $limit) {
            Logger::warning('Rate limit exceeded', ['identifier' => $identifier]);
            return false;
        }

        Cache::set($cache_key, $current + 1, 60);
        return true;
    }

    public static function getRemainingRequests($identifier = 'global') {
        $cache_key = Config::$RATE_LIMIT['cache_key'] . ':' . $identifier;
        $current = Cache::get($cache_key) ?: 0;
        $limit = Config::$RATE_LIMIT['requests_per_minute'];

        return max(0, $limit - $current);
    }
}

// ===== HTTP CLIENT CLASS =====
class HttpClient {
    private $timeout;

    public function __construct($timeout = 10) {
        $this->timeout = $timeout;
    }

    public function get($url, $headers = []) {
        $ch = curl_init($url);

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_HTTPHEADER => $headers,
            CURLOPT_SSL_VERIFYPEER => true,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_MAXREDIRS => 5
        ]);

        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error) {
            Logger::error('HTTP request failed', [
                'url' => $url,
                'error' => $error
            ]);
            return null;
        }

        if ($http_code !== 200) {
            Logger::warning('HTTP error', [
                'url' => $url,
                'code' => $http_code
            ]);
            return null;
        }

        return json_decode($response, true);
    }

    public function post($url, $data, $headers = []) {
        $ch = curl_init($url);

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($data),
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_HTTPHEADER => array_merge($headers, [
                'Content-Type: application/json'
            ]),
            CURLOPT_SSL_VERIFYPEER => true
        ]);

        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error) {
            Logger. error('HTTP POST failed', [
                'url' => $url,
                'error' => $error
            ]);
            return null;
        }

        return json_decode($response, true);
    }
}

// ===== BROKER ADAPTER CLASSES =====

/**
 * OANDA Forex Broker Adapter
 */
class OandaBroker {
    private $config;
    private $client;

    public function __construct($config = null) {
        $this->config = $config ?: Config::$OANDA;
        $this->client = new HttpClient($this->config['timeout']);
    }

    public function isEnabled() {
        return $this->config['enabled'];
    }

    /**
     * Get XAU/USD current price
     */
    public function getXAUUSDPrice() {
        $cache_key = 'oanda: xau_usd';
        $cached = Cache::get($cache_key);

        if ($cached) {
            Logger::debug('Returning cached OANDA price');
            return $cached;
        }

        if (!RateLimiter::isAllowed('oanda')) {
            Logger::warning('OANDA rate limit exceeded');
            return $cached ?: null;
        }

        $url = $this->config['endpoint'] . '/accounts/' . $this->config['account_id'] . '/pricing';
        $url .= '?instruments=' . $this->config['instruments'];

        $headers = [
            'Authorization: Bearer ' . $this->config['api_key'],
            'Accept-Datetime-Format: Unix'
        ];

        $response = $this->client->get($url, $headers);

        if (! $response || empty($response['prices'])) {
            Logger::error('OANDA API error', ['response' => $response]);
            return null;
        }

        $price_data = $response['prices'][0];

        $data = [
            'broker' => 'OANDA',
            'pair' => 'XAU/USD',
            'bid' => floatval($price_data['bids'][0]['price']),
            'ask' => floatval($price_data['asks'][0]['price']),
            'mid' => (floatval($price_data['bids'][0]['price']) + floatval($price_data['asks'][0]['price'])) / 2,
            'timestamp' => $price_data['time'],
            'liquidity' => 'HIGH',
            'status' => 'ACTIVE'
        ];

        Cache::set($cache_key, $data, 10);
        Logger::info('OANDA price retrieved', $data);

        return $data;
    }

    /**
     * Get historical data
     */
    public function getHistoricalData($granularity = 'D', $count = 100) {
        $url = $this->config['endpoint'] . '/instruments/' . $this->config['instruments'] . '/candles';
        $url .= '?granularity=' . $granularity . '&count=' .  $count;

        $headers = [
            'Authorization: Bearer ' . $this->config['api_key'],
            'Accept-Datetime-Format: Unix'
        ];

        $response = $this->client->get($url, $headers);

        if (!$response || empty($response['candles'])) {
            Logger:: error('OANDA historical data error');
            return [];
        }

        $candles = [];
        foreach ($response['candles'] as $candle) {
            $candles[] = [
                'time' => $candle['time'],
                'open' => floatval($candle['mid']['o']),
                'high' => floatval($candle['mid']['h']),
                'low' => floatval($candle['mid']['l']),
                'close' => floatval($candle['mid']['c']),
                'volume' => $candle['volume']
            ];
        }

        Logger::info('OANDA historical data retrieved', ['count' => count($candles)]);
        return $candles;
    }

    /**
     * Get account information
     */
    public function getAccountInfo() {
        $url = $this->config['endpoint'] . '/accounts/' . $this->config['account_id'];

        $headers = [
            'Authorization: Bearer ' . $this->config['api_key']
        ];

        $response = $this->client->get($url, $headers);

        if (!$response) {
            Logger::error('OANDA account info error');
            return null;
        }

        return [
            'broker' => 'OANDA',
            'account_id' => $response['account']['id'],
            'balance' => floatval($response['account']['balance']),
            'equity' => floatval($response['account']['balance']),
            'currency' => $response['account']['currency']
        ];
    }
}

/**
 * Alpha Vantage Forex Data Provider
 */
class AlphaVantageBroker {
    private $config;
    private $client;

    public function __construct($config = null) {
        $this->config = $config ?: Config::$ALPHA_VANTAGE;
        $this->client = new HttpClient($this->config['timeout']);
    }

    public function isEnabled() {
        return $this->config['enabled'];
    }

    /**
     * Get XAU/USD current price
     */
    public function getXAUUSDPrice() {
        $cache_key = 'alphavantage:xau_usd';
        $cached = Cache::get($cache_key);

        if ($cached) {
            Logger::debug('Returning cached Alpha Vantage price');
            return $cached;
        }

        if (!RateLimiter::isAllowed('alphavantage')) {
            Logger::warning('Alpha Vantage rate limit exceeded');
            return $cached ?: null;
        }

        $url = $this->config['endpoint'] . '?function=CURRENCY_EXCHANGE_RATE';
        $url .= '&from_currency=XAU&to_currency=USD&apikey=' . $this->config['api_key'];

        $response = $this->client->get($url);

        if (!$response || ! isset($response['Realtime Currency Exchange Rate'])) {
            Logger::error('Alpha Vantage API error');
            return null;
        }

        $rate_data = $response['Realtime Currency Exchange Rate'];

        $data = [
            'broker' => 'Alpha Vantage',
            'pair' => 'XAU/USD',
            'bid' => floatval($rate_data['5. Exchange Rate']) * 0.9999,
            'ask' => floatval($rate_data['5. Exchange Rate']) * 1.0001,
            'mid' => floatval($rate_data['5. Exchange Rate']),
            'timestamp' => date('c'),
            'liquidity' => 'HIGH',
            'status' => 'ACTIVE'
        ];

        Cache::set($cache_key, $data, 15);
        Logger::info('Alpha Vantage price retrieved', $data);

        return $data;
    }

    /**
     * Get daily time series
     */
    public function getDailyTimeSeries($symbols = 'XAUUSD', $outputsize = 'compact') {
        $url = $this->config['endpoint'] . '?function=FX_DAILY';
        $url .= '&from_symbol=XAU&to_symbol=USD';
        $url .= '&outputsize=' . $outputsize;
        $url .= '&apikey=' . $this->config['api_key'];

        $response = $this->client->get($url);

        if (!$response || !isset($response['Time Series FX (Daily)'])) {
            Logger::error('Alpha Vantage time series error');
            return [];
        }

        $timeseries = [];
        foreach ($response['Time Series FX (Daily)'] as $date => $data) {
            $timeseries[] = [
                'date' => $date,
                'open' => floatval($data['1. open']),
                'high' => floatval($data['2. high']),
                'low' => floatval($data['3. low']),
                'close' => floatval($data['4. close'])
            ];
        }

        Logger::info('Alpha Vantage time series retrieved', ['count' => count($timeseries)]);
        return $timeseries;
    }
}

/**
 * Finnhub Data Provider
 */
class FinnhubBroker {
    private $config;
    private $client;

    public function __construct($config = null) {
        $this->config = $config ?: Config::$FINNHUB;
        $this->client = new HttpClient($this->config['timeout']);
    }

    public function isEnabled() {
        return $this->config['enabled'];
    }

    /**
     * Get forex pair quote
     */
    public function getXAUUSDPrice() {
        $cache_key = 'finnhub:xau_usd';
        $cached = Cache::get($cache_key);

        if ($cached) {
            Logger::debug('Returning cached Finnhub price');
            return $cached;
        }

        if (!RateLimiter::isAllowed('finnhub')) {
            Logger:: warning('Finnhub rate limit exceeded');
            return $cached ?: null;
        }

        $url = $this->config['endpoint'] . '/forex/quote';
        $url .= '?symbol=OANDA: XAU_USD&token=' . $this->config['api_key'];

        $response = $this->client->get($url);

        if (!$response || empty($response['c'])) {
            Logger::error('Finnhub API error');
            return null;
        }

        $data = [
            'broker' => 'Finnhub',
            'pair' => 'XAU/USD',
            'bid' => floatval($response['c']) * 0.9999,
            'ask' => floatval($response['c']) * 1.0001,
            'mid' => floatval($response['c']),
            'high' => isset($response['h']) ? floatval($response['h']) : null,
            'low' => isset($response['l']) ? floatval($response['l']) : null,
            'timestamp' => isset($response['t']) ? date('c', $response['t']) : date('c'),
            'liquidity' => 'HIGH',
            'status' => 'ACTIVE'
        ];

        Cache:: set($cache_key, $data, 10);
        Logger::info('Finnhub price retrieved', $data);

        return $data;
    }

    /**
     * Get company candles
     */
    public function getCandles($symbol = 'OANDA: XAU_USD', $resolution = 'D', $count = 100) {
        $from = strtotime('-' . $count . ' days');
        $to = time();

        $url = $this->config['endpoint'] . '/forex/candle';
        $url .= '?symbol=' . $symbol;
        $url .= '&resolution=' . $resolution;
        $url .= '&from=' . $from .  '&to=' . $to;
        $url .= '&token=' . $this->config['api_key'];

        $response = $this->client->get($url);

        if (!$response || ! isset($response['o'])) {
            Logger::error('Finnhub candles error');
            return [];
        }

        $candles = [];
        for ($i = 0; $i < count($response['o']); $i++) {
            $candles[] = [
                'time' => $response['t'][$i],
                'open' => floatval($response['o'][$i]),
                'high' => floatval($response['h'][$i]),
                'low' => floatval($response['l'][$i]),
                'close' => floatval($response['c'][$i]),
                'volume' => isset($response['v'][$i]) ? $response['v'][$i] :  0
            ];
        }

        Logger::info('Finnhub candles retrieved', ['count' => count($candles)]);
        return $candles;
    }
}

/**
 * IEX Cloud Data Provider
 */
class IexCloudBroker {
    private $config;
    private $client;

    public function __construct($config = null) {
        $this->config = $config ?: Config::$IEX_CLOUD;
        $this->client = new HttpClient($this->config['timeout']);
    }

    public function isEnabled() {
        return $this->config['enabled'];
    }

    /**
     * Get forex rate
     */
    public function getXAUUSDPrice() {
        $cache_key = 'iex:xau_usd';
        $cached = Cache::get($cache_key);

        if ($cached) {
            Logger::debug('Returning cached IEX price');
            return $cached;
        }

        if (!RateLimiter::isAllowed('iex')) {
            Logger::warning('IEX rate limit exceeded');
            return $cached ?: null;
        }

        $url = $this->config['endpoint'] . '/fx/latest? symbols=XAUUSD&token=' . $this->config['api_key'];

        $response = $this->client->get($url);

        if (!$response || empty($response)) {
            Logger::error('IEX API error');
            return null;
        }

        $rate = $response[0];

        $data = [
            'broker' => 'IEX Cloud',
            'pair' => 'XAU/USD',
            'bid' => floatval($rate['rate']) * 0.9999,
            'ask' => floatval($rate['rate']) * 1.0001,
            'mid' => floatval($rate['rate']),
            'timestamp' => date('c'),
            'liquidity' => 'HIGH',
            'status' => 'ACTIVE'
        ];

        Cache::set($cache_key, $data, 10);
        Logger::info('IEX price retrieved', $data);

        return $data;
    }
}

// ===== BROKER AGGREGATOR CLASS =====
class BrokerAggregator {
    private $brokers = [];
    private $weights = [];

    public function __construct() {
        // Initialize enabled brokers
        if (true === (new OandaBroker())->isEnabled()) {
            $this->brokers['oanda'] = new OandaBroker();
            $this->weights['oanda'] = 0.4;  // 40% weight
        }

        if (true === (new AlphaVantageBroker())->isEnabled()) {
            $this->brokers['alphavantage'] = new AlphaVantageBroker();
            $this->weights['alphavantage'] = 0.3;  // 30% weight
        }

        if (true === (new FinnhubBroker())->isEnabled()) {
            $this->brokers['finnhub'] = new FinnhubBroker();
            $this->weights['finnhub'] = 0.2;  // 20% weight
        }

        if (true === (new IexCloudBroker())->isEnabled()) {
            $this->brokers['iex'] = new IexCloudBroker();
            $this->weights['iex'] = 0.1;  // 10% weight
        }

        Logger::info('Broker Aggregator initialized', [
            'active_brokers' => count($this->brokers),
            'brokers' => array_keys($this->brokers)
        ]);
    }

    /**
     * Get aggregated XAU/USD price from multiple sources
     */
    public function getAggregatedPrice() {
        $prices = [];
        $broker_data = [];

        foreach ($this->brokers as $name => $broker) {
            try {
                $price_data = $broker->getXAUUSDPrice();

                if ($price_data) {
                    $mid_price = $price_data['mid'];
                    $weight = $this->weights[$name];

                    $prices[$name] = [
                        'price' => $mid_price,
                        'weight' => $weight,
                        'weighted_price' => $mid_price * $weight
                    ];

                    $broker_data[] = $price_data;

                    Logger::debug('Broker price retrieved', [
                        'broker' => $name,
                        'price' => $mid_price,
                        'weight' => $weight
                    ]);
                }
            } catch (Exception $e) {
                Logger:: warning('Broker error', [
                    'broker' => $name,
                    'error' => $e->getMessage()
                ]);
            }
        }

        if (empty($prices)) {
            Logger::error('No broker data available');
            return null;
        }

        // Calculate weighted average
        $total_weight = array_sum(array_column($prices, 'weight'));
        $weighted_sum = array_sum(array_column($prices, 'weighted_price'));
        $aggregated_price = $weighted_sum / $total_weight;

        // Calculate statistics
        $all_prices = array_column($prices, 'price');
        $std_dev = $this->calculateStandardDeviation($all_prices);
        $spread = (max($all_prices) - min($all_prices)) / $aggregated_price * 100;

        $result = [
            'aggregated_price' => round($aggregated_price, 5),
            'bid' => round($aggregated_price * 0.9999, 5),
            'ask' => round($aggregated_price * 1.0001, 5),
            'high_price' => max($all_prices),
            'low_price' => min($all_prices),
            'standard_deviation' => $std_dev,
            'spread_percent' => $spread,
            'timestamp' => date('c'),
            'broker_count' => count($broker_data),
            'broker_prices' => $prices,
            'data_quality' => $this->assessDataQuality($prices),
            'liquidity' => $this->assessLiquidity($broker_data)
        ];

        Logger::info('Aggregated price calculated', [
            'price' => $result['aggregated_price'],
            'brokers' => count($broker_data)
        ]);

        return $result;
    }

    /**
     * Get historical data aggregated from brokers
     */
    public function getAggregatedHistoricalData($granularity = 'D', $count = 100) {
        $all_candles = [];

        if (isset($this->brokers['oanda'])) {
            try {
                $candles = $this->brokers['oanda']->getHistoricalData($granularity, $count);
                if ($candles) {
                    $all_candles['oanda'] = $candles;
                    Logger::info('OANDA historical data retrieved');
                }
            } catch (Exception $e) {
                Logger::warning('OANDA historical data error', ['error' => $e->getMessage()]);
            }
        }

        if (isset($this->brokers['finnhub'])) {
            try {
                $candles = $this->brokers['finnhub']->getCandles();
                if ($candles) {
                    $all_candles['finnhub'] = $candles;
                    Logger::info('Finnhub historical data retrieved');
                }
            } catch (Exception $e) {
                Logger::warning('Finnhub historical data error', ['error' => $e->getMessage()]);
            }
        }

        return $all_candles;
    }

    /**
     * Calculate standard deviation
     */
    private function calculateStandardDeviation($values) {
        $count = count($values);
        if ($count < 2) return 0;

        $mean = array_sum($values) / $count;
        $deviations = array_map(function($x) use ($mean) {
            return pow($x - $mean, 2);
        }, $values);

        return sqrt(array_sum($deviations) / $count);
    }

    /**
     * Assess data quality based on broker consistency
     */
    private function assessDataQuality($prices) {
        $prices_list = array_column($prices, 'price');
        $std_dev = $this->calculateStandardDeviation($prices_list);
        $mean = array_sum($prices_list) / count($prices_list);
        $cv = ($std_dev / $mean) * 100;  // Coefficient of variation

        if ($cv < 0.1) return 'EXCELLENT';
        if ($cv < 0.5) return 'GOOD';
        if ($cv < 1.0) return 'FAIR';
        return 'POOR';
    }

    /**
     * Assess market liquidity
     */
    private function assessLiquidity($broker_data) {
        $avg_spread = array_sum(array_column($broker_data, 'spread_percent', null)) / count($broker_data);

        if ($avg_spread < 0.001) return 'ULTRA_HIGH';
        if ($avg_spread < 0.01) return 'VERY_HIGH';
        if ($avg_spread < 0.05) return 'HIGH';
        if ($avg_spread < 0.1) return 'MEDIUM';
        return 'LOW';
    }

    /**
     * Get broker status
     */
    public function getBrokerStatus() {
        $status = [];

        foreach ($this->brokers as $name => $broker) {
            try {
                $price_data = $broker->getXAUUSDPrice();
                $status[$name] = [
                    'status' => $price_data ? 'ACTIVE' : 'INACTIVE',
                    'last_update' => $price_data ? $price_data['timestamp'] : null,
                    'weight' => $this->weights[$name]
                ];
            } catch (Exception $e) {
                $status[$name] = [
                    'status' => 'ERROR',
                    'error' => $e->getMessage(),
                    'weight' => $this->weights[$name]
                ];
            }
        }

        return $status;
    }
}

// ===== API ENDPOINT HANDLERS =====

// Set JSON header
header('Content-Type: application/json');

// Get request method and path
$method = $_SERVER['REQUEST_METHOD'];
$path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
$path = preg_replace('/^.*\/api\//', '', $path);

try {
    $aggregator = new BrokerAggregator();

    // Router
    if ('GET' === $method) {
        if ('price' === $path || '/price' === $path) {
            // Get aggregated price
            $price = $aggregator->getAggregatedPrice();

            if ($price) {
                http_response_code(200);
                echo json_encode([
                    'success' => true,
                    'data' => $price
                ]);
            } else {
                http_response_code(503);
                echo json_encode([
                    'success' => false,
                    'error' => 'No broker data available',
                    'timestamp' => date('c')
                ]);
            }
        } elseif ('historical' === $path || '/historical' === $path) {
            // Get historical data
            $granularity = $_GET['granularity'] ?? 'D';
            $count = $_GET['count'] ?? 100;

            $data = $aggregator->getAggregatedHistoricalData($granularity, $count);

            if ($data) {
                http_response_code(200);
                echo json_encode([
                    'success' => true,
                    'data' => $data,
                    'timestamp' => date('c')
                ]);
            } else {
                http_response_code(503);
                echo json_encode([
                    'success' => false,
                    'error' => 'No historical data available'
                ]);
            }
        } elseif ('status' === $path || '/status' === $path) {
            // Get broker status
            $status = $aggregator->getBrokerStatus();

            http_response_code(200);
            echo json_encode([
                'success' => true,
                'data' => $status,
                'timestamp' => date('c')
            ]);
        } elseif ('health' === $path || '/health' === $path) {
            // Health check
            http_response_code(200);
            echo json_encode([
                'success' => true,
                'status' => 'OK',
                'service' => 'Broker Data Aggregation API',
                'timestamp' => date('c')
            ]);
        } else {
            http_response_code(404);
            echo json_encode([
                'success' => false,
                'error' => 'Endpoint not found'
            ]);
        }
    } else {
        http_response_code(405);
        echo json_encode([
            'success' => false,
            'error' => 'Method not allowed'
        ]);
    }

} catch (Exception $e) {
    Logger::error('API error', [
        'path' => $path,
        'error' => $e->getMessage()
    ]);

    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Internal server error',
        'message' => $e->getMessage()
    ]);
}
?>
