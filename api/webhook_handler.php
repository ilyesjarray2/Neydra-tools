<?php
/**
 * Webhook Handler
 * Real-Time Data Streaming & Event Processing
 * NEYDRA Platform - Enterprise Grade
 * Founder & CEO: Ilyes Jarray
 * © 2025 - All Rights Reserved
 */

// ===== ERROR HANDLING & LOGGING =====
error_reporting(E_ALL);
ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', __DIR__ . '/logs/webhooks.log');

if (! is_dir(__DIR__ . '/logs')) {
    mkdir(__DIR__ . '/logs', 0755, true);
}

date_default_timezone_set('UTC');

// ===== CONFIGURATION =====
class WebhookConfig {
    // Webhook Security
    public static $SECURITY = [
        'enable_signature_verification' => true,
        'secret_keys' => [
            'oanda' => getenv('OANDA_WEBHOOK_SECRET') ?: 'oanda_secret_key',
            'broker' => getenv('BROKER_WEBHOOK_SECRET') ?: 'broker_secret_key',
            'system' => getenv('SYSTEM_WEBHOOK_SECRET') ?: 'system_secret_key'
        ],
        'allowed_ips' => [
            '127.0.0.1',
            'localhost',
            getenv('WEBHOOK_ALLOWED_IPS') ?  explode(',', getenv('WEBHOOK_ALLOWED_IPS')) : []
        ]
    ];

    // Queue Configuration
    public static $QUEUE = [
        'enabled' => true,
        'max_retries' => 3,
        'retry_delay' => 5, // seconds
        'dir' => __DIR__ . '/queue'
    ];

    // Event Types
    public static $EVENTS = [
        'PRICE_UPDATE' => 'price. updated',
        'TRADE_SIGNAL' => 'signal.generated',
        'ALERT' => 'alert.triggered',
        'ERROR' => 'error.occurred',
        'BROKER_DATA' => 'broker.data. received',
        'PREDICTION' => 'prediction.generated',
        'MARKET_CLOSE' => 'market.closed',
        'MARKET_OPEN' => 'market.opened'
    ];

    // WebSocket Configuration
    public static $WEBSOCKET = [
        'enabled' => true,
        'port' => 8080,
        'host' => '0.0.0.0',
        'max_connections' => 1000,
        'ping_interval' => 30 // seconds
    ];

    // Database Configuration
    public static $DATABASE = [
        'enabled' => true,
        'host' => getenv('DB_HOST') ?: 'localhost',
        'port' => getenv('DB_PORT') ?: 5432,
        'name' => getenv('DB_NAME') ?: 'neydra',
        'user' => getenv('DB_USER') ?: 'postgres',
        'password' => getenv('DB_PASSWORD') ?: 'password'
    ];

    // Notification Endpoints
    public static $NOTIFICATIONS = [
        'slack' => getenv('SLACK_WEBHOOK_URL') ?: null,
        'email' => getenv('NOTIFICATION_EMAIL') ?: null,
        'telegram' => getenv('TELEGRAM_BOT_TOKEN') ?: null
    ];
}

// ===== LOGGER CLASS =====
class WebhookLogger {
    private static $log_file = __DIR__ . '/logs/webhooks.log';
    private static $event_file = __DIR__ . '/logs/events.log';

    public static function log($level, $message, $context = []) {
        $timestamp = date('Y-m-d H:i:s');
        $context_str = ! empty($context) ? json_encode($context) : '';
        $log_message = "[{$timestamp}] [{$level}] {$message} {$context_str}\n";

        error_log($log_message, 3, self::$log_file);
    }

    public static function logEvent($event_type, $data) {
        $timestamp = date('c');
        $event = [
            'timestamp' => $timestamp,
            'type' => $event_type,
            'data' => $data
        ];

        $event_line = json_encode($event) . "\n";
        error_log($event_line, 3, self::$event_file);
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

// ===== SIGNATURE VERIFICATION CLASS =====
class SignatureVerifier {
    public static function verify($payload, $signature, $source) {
        if (!WebhookConfig::$SECURITY['enable_signature_verification']) {
            return true;
        }

        if (!isset(WebhookConfig::$SECURITY['secret_keys'][$source])) {
            WebhookLogger::warning('Unknown webhook source', ['source' => $source]);
            return false;
        }

        $secret = WebhookConfig::$SECURITY['secret_keys'][$source];
        $expected_signature = hash_hmac('sha256', $payload, $secret);

        $is_valid = hash_equals($expected_signature, $signature);

        if (!$is_valid) {
            WebhookLogger:: warning('Signature verification failed', [
                'source' => $source,
                'expected' => $expected_signature,
                'received' => $signature
            ]);
        }

        return $is_valid;
    }

    public static function generateSignature($payload, $source) {
        if (!isset(WebhookConfig::$SECURITY['secret_keys'][$source])) {
            return null;
        }

        $secret = WebhookConfig::$SECURITY['secret_keys'][$source];
        return hash_hmac('sha256', $payload, $secret);
    }
}

// ===== IP VALIDATION CLASS =====
class IpValidator {
    public static function isAllowed($ip) {
        $allowed_ips = WebhookConfig::$SECURITY['allowed_ips'];

        // Flatten array if nested
        $allowed_ips = array_merge(... array_filter($allowed_ips, 'is_array'), array_filter($allowed_ips, 'is_string'));

        return in_array($ip, $allowed_ips) || empty($allowed_ips);
    }

    public static function getClientIp() {
        if (!empty($_SERVER['HTTP_CLIENT_IP'])) {
            return $_SERVER['HTTP_CLIENT_IP'];
        } elseif (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
            // Handle multiple IPs in X-Forwarded-For
            $ips = explode(',', $_SERVER['HTTP_X_FORWARDED_FOR']);
            return trim($ips[0]);
        } else {
            return $_SERVER['REMOTE_ADDR'];
        }
    }
}

// ===== QUEUE MANAGER CLASS =====
class QueueManager {
    private static $queue_dir;

    public static function init() {
        self::$queue_dir = WebhookConfig::$QUEUE['dir'];

        if (!is_dir(self::$queue_dir)) {
            mkdir(self::$queue_dir, 0755, true);
        }
    }

    public static function enqueue($event_type, $data) {
        self::init();

        $queue_item = [
            'id' => uniqid('event_', true),
            'event_type' => $event_type,
            'data' => $data,
            'timestamp' => date('c'),
            'retry_count' => 0,
            'max_retries' => WebhookConfig::$QUEUE['max_retries'],
            'status' => 'PENDING'
        ];

        $filename = self::$queue_dir . '/' . $queue_item['id'] . '.queue';
        file_put_contents($filename, json_encode($queue_item, JSON_PRETTY_PRINT));

        WebhookLogger::info('Event queued', [
            'id' => $queue_item['id'],
            'type' => $event_type
        ]);

        return $queue_item['id'];
    }

    public static function dequeue($id) {
        self::init();

        $filename = self::$queue_dir . '/' . $id . '.queue';

        if (file_exists($filename)) {
            unlink($filename);
            WebhookLogger::info('Event dequeued', ['id' => $id]);
            return true;
        }

        return false;
    }

    public static function getPending() {
        self::init();

        $pending = [];
        $files = glob(self::$queue_dir . '/*.queue');

        foreach ($files as $file) {
            $data = json_decode(file_get_contents($file), true);
            if ($data['status'] === 'PENDING') {
                $pending[] = $data;
            }
        }

        return $pending;
    }

    public static function retry($id) {
        self::init();

        $filename = self::$queue_dir . '/' . $id . '.queue';

        if (! file_exists($filename)) {
            return false;
        }

        $item = json_decode(file_get_contents($filename), true);
        $item['retry_count']++;

        if ($item['retry_count'] >= $item['max_retries']) {
            $item['status'] = 'FAILED';
            WebhookLogger::error('Max retries exceeded', [
                'id' => $id,
                'retry_count' => $item['retry_count']
            ]);
        }

        file_put_contents($filename, json_encode($item, JSON_PRETTY_PRINT));
        return true;
    }
}

// ===== EVENT PROCESSOR CLASS =====
class EventProcessor {
    private $db;

    public function __construct() {
        if (WebhookConfig::$DATABASE['enabled']) {
            $this->connectDatabase();
        }
    }

    private function connectDatabase() {
        try {
            $dsn = sprintf(
                'pgsql:host=%s;port=%d;dbname=%s',
                WebhookConfig::$DATABASE['host'],
                WebhookConfig::$DATABASE['port'],
                WebhookConfig::$DATABASE['name']
            );

            $this->db = new PDO(
                $dsn,
                WebhookConfig::$DATABASE['user'],
                WebhookConfig::$DATABASE['password'],
                [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]
            );

            WebhookLogger::info('Database connection established');
        } catch (Exception $e) {
            WebhookLogger::error('Database connection failed', [
                'error' => $e->getMessage()
            ]);
        }
    }

    public function processEvent($event_type, $data) {
        WebhookLogger::info('Processing event', [
            'type' => $event_type,
            'data' => $data
        ]);

        // Log event
        WebhookLogger:: logEvent($event_type, $data);

        // Store in database if enabled
        if ($this->db) {
            $this->storeEvent($event_type, $data);
        }

        // Process based on type
        switch ($event_type) {
            case WebhookConfig::$EVENTS['PRICE_UPDATE']:
                return $this->handlePriceUpdate($data);

            case WebhookConfig::$EVENTS['TRADE_SIGNAL']:
                return $this->handleTradeSignal($data);

            case WebhookConfig::$EVENTS['ALERT']:
                return $this->handleAlert($data);

            case WebhookConfig::$EVENTS['BROKER_DATA']:
                return $this->handleBrokerData($data);

            case WebhookConfig::$EVENTS['PREDICTION']:
                return $this->handlePrediction($data);

            default:
                return ['status' => 'processed', 'type' => $event_type];
        }
    }

    private function storeEvent($event_type, $data) {
        try {
            $sql = "INSERT INTO events (type, data, created_at) VALUES (?, ?, NOW())";
            $stmt = $this->db->prepare($sql);
            $stmt->execute([
                $event_type,
                json_encode($data)
            ]);

            WebhookLogger::debug('Event stored in database', [
                'type' => $event_type
            ]);
        } catch (Exception $e) {
            WebhookLogger::error('Failed to store event', [
                'type' => $event_type,
                'error' => $e->getMessage()
            ]);
        }
    }

    private function handlePriceUpdate($data) {
        WebhookLogger::info('Handling price update', $data);

        // Broadcast to WebSocket clients
        $this->broadcastToClients('price_update', $data);

        // Store price history
        if ($this->db) {
            try {
                $sql = "INSERT INTO price_history (pair, bid, ask, mid, timestamp) VALUES (?, ?, ?, ?, NOW())";
                $stmt = $this->db->prepare($sql);
                $stmt->execute([
                    $data['pair'] ??  'XAU/USD',
                    $data['bid'] ?? null,
                    $data['ask'] ?? null,
                    $data['mid'] ?? null
                ]);
            } catch (Exception $e) {
                WebhookLogger:: error('Failed to store price', ['error' => $e->getMessage()]);
            }
        }

        // Send notifications if configured
        $this->notifySubscribers('price_update', $data);

        return ['status' => 'price_update_processed'];
    }

    private function handleTradeSignal($data) {
        WebhookLogger::info('Handling trade signal', $data);

        // Broadcast signal
        $this->broadcastToClients('trade_signal', $data);

        // Store signal
        if ($this->db) {
            try {
                $sql = "INSERT INTO signals (type, price, confidence, timestamp) VALUES (?, ?, ?, NOW())";
                $stmt = $this->db->prepare($sql);
                $stmt->execute([
                    $data['signal'] ?? 'HOLD',
                    $data['price'] ?? null,
                    $data['confidence'] ?? null
                ]);
            } catch (Exception $e) {
                WebhookLogger::error('Failed to store signal', ['error' => $e->getMessage()]);
            }
        }

        // Send alerts for strong signals
        if (isset($data['confidence']) && $data['confidence'] > 0.85) {
            $this->sendAlert('Strong Trading Signal', $data);
        }

        return ['status' => 'signal_processed'];
    }

    private function handleAlert($data) {
        WebhookLogger::info('Handling alert', $data);

        // Broadcast alert
        $this->broadcastToClients('alert', $data);

        // Send notifications
        $this->sendAlert($data['title'] ?? 'Alert', $data);

        // Store alert
        if ($this->db) {
            try {
                $sql = "INSERT INTO alerts (title, message, severity, timestamp) VALUES (?, ?, ?, NOW())";
                $stmt = $this->db->prepare($sql);
                $stmt->execute([
                    $data['title'] ?? 'Alert',
                    $data['message'] ?? '',
                    $data['severity'] ?? 'INFO'
                ]);
            } catch (Exception $e) {
                WebhookLogger::error('Failed to store alert', ['error' => $e->getMessage()]);
            }
        }

        return ['status' => 'alert_processed'];
    }

    private function handleBrokerData($data) {
        WebhookLogger:: info('Handling broker data', $data);

        // Broadcast to clients
        $this->broadcastToClients('broker_data', $data);

        // Store broker data
        if ($this->db) {
            try {
                $sql = "INSERT INTO broker_data (broker, pair, price, timestamp) VALUES (?, ?, ?, NOW())";
                $stmt = $this->db->prepare($sql);
                $stmt->execute([
                    $data['broker'] ?? 'UNKNOWN',
                    $data['pair'] ?? 'XAU/USD',
                    $data['price'] ?? null
                ]);
            } catch (Exception $e) {
                WebhookLogger::error('Failed to store broker data', ['error' => $e->getMessage()]);
            }
        }

        return ['status' => 'broker_data_processed'];
    }

    private function handlePrediction($data) {
        WebhookLogger::info('Handling prediction', $data);

        // Broadcast prediction
        $this->broadcastToClients('prediction', $data);

        // Store prediction
        if ($this->db) {
            try {
                $sql = "INSERT INTO predictions (direction, confidence, target_price, timestamp) VALUES (?, ?, ?, NOW())";
                $stmt = $this->db->prepare($sql);
                $stmt->execute([
                    $data['direction'] ?? 'NEUTRAL',
                    $data['confidence'] ?? null,
                    $data['target_price'] ?? null
                ]);
            } catch (Exception $e) {
                WebhookLogger::error('Failed to store prediction', ['error' => $e->getMessage()]);
            }
        }

        return ['status' => 'prediction_processed'];
    }

    private function broadcastToClients($event_type, $data) {
        // This would connect to WebSocket server to broadcast
        WebhookLogger::debug('Broadcasting to clients', [
            'type' => $event_type,
            'client_count' => 0  // Would be dynamic
        ]);
    }

    private function notifySubscribers($event_type, $data) {
        // Send to Slack
        if (WebhookConfig::$NOTIFICATIONS['slack']) {
            $this->sendSlackNotification($event_type, $data);
        }

        // Send email
        if (WebhookConfig::$NOTIFICATIONS['email']) {
            $this->sendEmailNotification($event_type, $data);
        }

        // Send Telegram
        if (WebhookConfig::$NOTIFICATIONS['telegram']) {
            $this->sendTelegramNotification($event_type, $data);
        }
    }

    private function sendAlert($title, $data) {
        WebhookLogger::info('Sending alert', ['title' => $title]);

        $this->sendSlackNotification('alert', [
            'title' => $title,
            'data' => $data
        ]);

        $this->sendEmailNotification('alert', [
            'title' => $title,
            'data' => $data
        ]);

        $this->sendTelegramNotification('alert', [
            'title' => $title,
            'data' => $data
        ]);
    }

    private function sendSlackNotification($type, $data) {
        if (!WebhookConfig::$NOTIFICATIONS['slack']) {
            return;
        }

        $webhook_url = WebhookConfig::$NOTIFICATIONS['slack'];

        $message = [
            'text' => "🔔 NEYDRA Event: {$type}",
            'blocks' => [
                [
                    'type' => 'section',
                    'text' => [
                        'type' => 'mrkdwn',
                        'text' => "*Event Type:* {$type}\n*Timestamp:* " . date('c')
                    ]
                ],
                [
                    'type' => 'section',
                    'text' => [
                        'type' => 'mrkdwn',
                        'text' => '```' . json_encode($data, JSON_PRETTY_PRINT) . '```'
                    ]
                ]
            ]
        ];

        $ch = curl_init($webhook_url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($message),
            CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
            CURLOPT_TIMEOUT => 10
        ]);

        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($http_code === 200) {
            WebhookLogger:: debug('Slack notification sent');
        } else {
            WebhookLogger::warning('Slack notification failed', ['code' => $http_code]);
        }
    }

    private function sendEmailNotification($type, $data) {
        if (!WebhookConfig::$NOTIFICATIONS['email']) {
            return;
        }

        $email = WebhookConfig::$NOTIFICATIONS['email'];
        $subject = "NEYDRA Alert: {$type}";
        $message = "Event Type:  {$type}\n\n";
        $message .= json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
        $message .= "\n\nTimestamp: " . date('c');

        $headers = "From: alerts@neydra.io\r\n";
        $headers . = "Content-Type: text/plain; charset=UTF-8\r\n";

        $result = mail($email, $subject, $message, $headers);

        if ($result) {
            WebhookLogger::debug('Email notification sent', ['to' => $email]);
        } else {
            WebhookLogger::warning('Email notification failed');
        }
    }

    private function sendTelegramNotification($type, $data) {
        if (!WebhookConfig:: $NOTIFICATIONS['telegram']) {
            return;
        }

        $bot_token = WebhookConfig::$NOTIFICATIONS['telegram'];
        $chat_id = getenv('TELEGRAM_CHAT_ID') ?: '';

        if (!$chat_id) {
            return;
        }

        $message = "🔔 *NEYDRA Alert*\n\n";
        $message .= "*Event: * {$type}\n";
        $message .= "*Time:* " . date('Y-m-d H:i:s') . "\n";
        $message .= "*Data:* \n```\n" . json_encode($data, JSON_PRETTY_PRINT) . "\n```";

        $url = "https://api.telegram.org/bot{$bot_token}/sendMessage";
        $params = [
            'chat_id' => $chat_id,
            'text' => $message,
            'parse_mode' => 'Markdown'
        ];

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => http_build_query($params),
            CURLOPT_TIMEOUT => 10
        ]);

        $response = curl_exec($ch);
        curl_close($ch);

        WebhookLogger::debug('Telegram notification sent');
    }
}

// ===== WEBHOOK HANDLER CLASS =====
class WebhookHandler {
    private $processor;
    private $request_body;
    private $source;

    public function __construct() {
        $this->processor = new EventProcessor();
        $this->request_body = file_get_contents('php://input');
    }

    public function handle() {
        // Validate IP
        $client_ip = IpValidator::getClientIp();
        if (!IpValidator::isAllowed($client_ip)) {
            WebhookLogger::warning('IP not allowed', ['ip' => $client_ip]);
            http_response_code(403);
            return $this->jsonResponse([
                'success' => false,
                'error' => 'IP not allowed'
            ]);
        }

        // Get source from header
        $this->source = $_SERVER['HTTP_X_WEBHOOK_SOURCE'] ?? 'unknown';

        // Verify signature
        $signature = $_SERVER['HTTP_X_SIGNATURE'] ?? '';
        if (!SignatureVerifier::verify($this->request_body, $signature, $this->source)) {
            WebhookLogger::warning('Signature verification failed', [
                'source' => $this->source
            ]);
            http_response_code(401);
            return $this->jsonResponse([
                'success' => false,
                'error' => 'Signature verification failed'
            ]);
        }

        // Parse payload
        $payload = json_decode($this->request_body, true);

        if (!$payload) {
            WebhookLogger:: error('Invalid JSON payload');
            http_response_code(400);
            return $this->jsonResponse([
                'success' => false,
                'error' => 'Invalid JSON payload'
            ]);
        }

        // Extract event type and data
        $event_type = $payload['event_type'] ?? $payload['type'] ?? 'UNKNOWN';
        $data = $payload['data'] ?? $payload;

        WebhookLogger::info('Webhook received', [
            'source' => $this->source,
            'event_type' => $event_type,
            'ip' => $client_ip
        ]);

        // Queue event for processing
        $queue_id = QueueManager::enqueue($event_type, $data);

        // Process event asynchronously
        try {
            $result = $this->processor->processEvent($event_type, $data);

            QueueManager::dequeue($queue_id);

            http_response_code(200);
            return $this->jsonResponse([
                'success' => true,
                'message' => 'Event processed successfully',
                'queue_id' => $queue_id,
                'event_type' => $event_type,
                'timestamp' => date('c')
            ]);
        } catch (Exception $e) {
            WebhookLogger::error('Event processing error', [
                'error' => $e->getMessage(),
                'queue_id' => $queue_id
            ]);

            QueueManager::retry($queue_id);

            http_response_code(500);
            return $this->jsonResponse([
                'success' => false,
                'error' => 'Event processing failed',
                'queue_id' => $queue_id
            ]);
        }
    }

    private function jsonResponse($data) {
        header('Content-Type: application/json');
        echo json_encode($data);
        exit;
    }
}

// ===== API ENDPOINTS =====

header('Content-Type: application/json');

$method = $_SERVER['REQUEST_METHOD'];
$path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
$path = preg_replace('/^.*\/webhook/', '', $path) ?: '/';

try {
    if ('POST' === $method && ('/' === $path || '/webhook' === $path)) {
        // Main webhook endpoint
        $handler = new WebhookHandler();
        $handler->handle();

    } elseif ('GET' === $method && '/status' === $path) {
        // Status endpoint
        http_response_code(200);
        echo json_encode([
            'success' => true,
            'status' => 'OPERATIONAL',
            'service' => 'Webhook Handler',
            'timestamp' => date('c'),
            'queue_pending' => count(QueueManager::getPending())
        ]);

    } elseif ('GET' === $method && '/queue' === $path) {
        // Queue status endpoint
        http_response_code(200);
        echo json_encode([
            'success' => true,
            'pending_events' => QueueManager::getPending(),
            'timestamp' => date('c')
        ]);

    } elseif ('POST' === $method && '/test' === $path) {
        // Test webhook
        $payload = [
            'event_type' => 'TEST_EVENT',
            'data' => [
                'message' => 'Test webhook payload',
                'timestamp' => date('c')
            ]
        ];

        $body = json_encode($payload);
        $signature = SignatureVerifier::generateSignature($body, 'system');

        $handler = new WebhookHandler();
        $_SERVER['HTTP_X_WEBHOOK_SOURCE'] = 'system';
        $_SERVER['HTTP_X_SIGNATURE'] = $signature;

        ob_start();
        // Simulate request
        echo json_encode([
            'success' => true,
            'message' => 'Test webhook created',
            'payload' => $payload,
            'signature' => $signature
        ]);
        ob_end_flush();

    } else {
        http_response_code(404);
        echo json_encode([
            'success' => false,
            'error' => 'Endpoint not found'
        ]);
    }

} catch (Exception $e) {
    WebhookLogger::error('Webhook handler error', [
        'error' => $e->getMessage(),
        'file' => $e->getFile(),
        'line' => $e->getLine()
    ]);

    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Internal server error',
        'message' => $e->getMessage()
    ]);
}
?>