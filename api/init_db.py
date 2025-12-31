"""
Database Initialization Script
XAU/USD Predictive Analytics Engine - NEYDRA Platform
Handles database setup, migrations, and initial data loading
Founder & CEO: Ilyes Jarray
© 2025 - All Rights Reserved
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys. path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2 import sql, errors
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy. pool import NullPool

# ===== LOGGING CONFIGURATION =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database_init.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===== DATABASE INITIALIZATION CLASS =====
class DatabaseInitializer:
    """
    Handles database initialization, schema creation, and data seeding
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'neydra',
        user: str = 'postgres',
        password: str = 'password',
        create_db: bool = True,
        schema_file: Optional[str] = None
    ):
        """
        Initialize database connection parameters
        
        Args:
            host:  Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            create_db: Whether to create database if it doesn't exist
            schema_file: Path to SQL schema file
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.create_db = create_db
        self.schema_file = schema_file or Path(__file__).parent / 'schema.sql'
        
        # Connection URLs
        self.master_url = f"postgresql://{user}:{password}@{host}:{port}/postgres"
        self.db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        self.engine = None
        self.session_factory = None
        
        logger.info(f"🔧 Database Initializer created for {database}@{host}:{port}")

    def check_postgresql_connection(self) -> bool:
        """
        Check if PostgreSQL server is accessible
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info("🔍 Checking PostgreSQL connection...")
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database='postgres'
            )
            conn.close()
            logger.info("✅ PostgreSQL server is accessible")
            return True
        except psycopg2.OperationalError as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error:  {str(e)}")
            return False

    def create_database(self) -> bool:
        """
        Create database if it doesn't exist
        
        Returns: 
            bool: True if database created or already exists
        """
        if not self.create_db:
            logger.info("⏭️  Skipping database creation (create_db=False)")
            return True

        try:
            logger.info(f"📦 Creating database '{self.database}' if it doesn't exist...")
            
            # Connect to master database
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database='postgres'
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;",
                (self.database,)
            )
            
            if cursor.fetchone() is None:
                logger.info(f"Creating database {self.database}...")
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.database)
                ))
                logger.info(f"✅ Database '{self.database}' created successfully")
            else:
                logger.info(f"✅ Database '{self.database}' already exists")

            cursor.close()
            conn.close()
            return True

        except psycopg2.Error as e:
            logger.error(f"❌ Database creation failed: {str(e)}")
            return False

    def execute_schema(self) -> bool:
        """
        Execute SQL schema file to create tables and structures
        
        Returns:
            bool: True if schema execution successful
        """
        try: 
            if not Path(self.schema_file).exists():
                logger.error(f"❌ Schema file not found: {self.schema_file}")
                return False

            logger. info(f"📋 Executing schema from {self.schema_file}...")

            # Read schema file
            with open(self. schema_file, 'r') as f:
                schema_sql = f.read()

            # Connect and execute
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            cursor = conn.cursor()

            # Split by semicolon and execute each statement
            statements = schema_sql.split(';')
            for statement in statements:
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except psycopg2.Error as e:
                        logger. warning(f"⚠️  Error executing statement:  {str(e)}")
                        # Continue with other statements
                        continue

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("✅ Schema executed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Schema execution failed: {str(e)}")
            return False

    def create_engine(self) -> bool:
        """
        Create SQLAlchemy engine for database operations
        
        Returns:
            bool: True if engine created successfully
        """
        try:
            logger.info("🔌 Creating SQLAlchemy engine...")
            
            self.engine = create_engine(
                self.db_url,
                poolclass=NullPool,
                echo=False,
                connect_args={
                    'connect_timeout': 10,
                    'options': '-c default_transaction_isolation=read_committed'
                }
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self.session_factory = sessionmaker(bind=self.engine)
            logger.info("✅ SQLAlchemy engine created successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Engine creation failed: {str(e)}")
            return False

    def seed_initial_data(self) -> bool:
        """
        Seed database with initial data
        
        Returns:
            bool: True if seeding successful
        """
        try:
            logger.info("🌱 Seeding initial data...")

            if not self.session_factory:
                logger.error("❌ Session factory not initialized")
                return False

            session = self.session_factory()

            try:
                # Check if data already exists
                result = session.execute(text("SELECT COUNT(*) FROM users"))
                if result.scalar() > 0:
                    logger.info("✅ Data already exists, skipping seed")
                    session.close()
                    return True

                # Seed users
                logger.info("Adding initial users...")
                self._seed_users(session)

                # Seed market data
                logger.info("Adding sample market data...")
                self._seed_market_data(session)

                # Seed broker data
                logger.info("Adding sample broker data...")
                self._seed_broker_data(session)

                # Seed predictions
                logger.info("Adding sample predictions...")
                self._seed_predictions(session)

                # Seed signals
                logger.info("Adding sample signals...")
                self._seed_signals(session)

                session. commit()
                logger.info("✅ Initial data seeded successfully")
                return True

            except Exception as e: 
                session.rollback()
                logger. error(f"❌ Seeding failed: {str(e)}")
                return False
            finally:
                session.close()

        except Exception as e:
            logger.error(f"❌ Seed initialization failed: {str(e)}")
            return False

    def _seed_users(self, session:  Session):
        """Seed initial users"""
        from bcrypt import hashpw, gensalt

        users_data = [
            {
                'username': 'admin',
                'email': 'admin@neydra.io',
                'full_name': 'Administrator',
                'is_admin': True
            },
            {
                'username': 'trader',
                'email': 'trader@neydra.io',
                'full_name': 'Demo Trader',
                'is_admin': False
            },
            {
                'username': 'analyst',
                'email': 'analyst@neydra.io',
                'full_name': 'Data Analyst',
                'is_admin': False
            }
        ]

        for user_data in users_data:
            password_hash = hashpw(b'password123', gensalt()).decode('utf-8')
            
            insert_stmt = text("""
                INSERT INTO users (username, email, full_name, password_hash, is_admin, is_active)
                VALUES (: username, :email, :full_name, :password_hash, : is_admin, true)
                ON CONFLICT (username) DO NOTHING
            """)
            
            session.execute(insert_stmt, {
                'username': user_data['username'],
                'email':  user_data['email'],
                'full_name': user_data['full_name'],
                'password_hash': password_hash,
                'is_admin': user_data['is_admin']
            })

        logger.info("✅ Users seeded")

    def _seed_market_data(self, session: Session):
        """Seed sample market data"""
        base_price = 2030. 50
        
        for i in range(30):
            date = datetime.now() - timedelta(days=30-i)
            price_change = (i - 15) * 2  # Price trend
            
            insert_stmt = text("""
                INSERT INTO market_data 
                (pair, open_price, high_price, low_price, close_price, volume, 
                 bid_price, ask_price, mid_price, source, data_quality, timestamp)
                VALUES (:pair, :open_price, :high_price, :low_price, : close_price, 
                        :volume, :bid_price, :ask_price, :mid_price, :source, : data_quality, :timestamp)
            """)
            
            close_price = base_price + price_change
            
            session.execute(insert_stmt, {
                'pair': 'XAU/USD',
                'open_price': close_price - 5,
                'high_price':  close_price + 10,
                'low_price':  close_price - 10,
                'close_price': close_price,
                'volume': 1000000 + (i * 10000),
                'bid_price':  close_price - 0.02,
                'ask_price': close_price + 0.02,
                'mid_price': close_price,
                'source':  'OANDA',
                'data_quality':  'EXCELLENT',
                'timestamp': date
            })

        logger.info("✅ Market data seeded")

    def _seed_broker_data(self, session: Session):
        """Seed sample broker data"""
        brokers = ['OANDA', 'Alpha Vantage', 'Finnhub']
        base_price = 2030.50

        for i in range(10):
            for broker in brokers:
                insert_stmt = text("""
                    INSERT INTO broker_data 
                    (broker_name, pair, bid, ask, mid, bid_volume, ask_volume, spread_percent, 
                     data_quality, timestamp)
                    VALUES (:broker_name, : pair, :bid, :ask, :mid, :bid_volume, :ask_volume, 
                            :spread_percent, :data_quality, :timestamp)
                """)
                
                price = base_price + (i * 5)
                
                session.execute(insert_stmt, {
                    'broker_name': broker,
                    'pair': 'XAU/USD',
                    'bid': price - 0.02,
                    'ask': price + 0.02,
                    'mid': price,
                    'bid_volume': 500000,
                    'ask_volume':  500000,
                    'spread_percent': 0.0001,
                    'data_quality': 'GOOD',
                    'timestamp': datetime.now() - timedelta(minutes=i*10)
                })

        logger.info("✅ Broker data seeded")

    def _seed_predictions(self, session: Session):
        """Seed sample predictions"""
        directions = ['UP', 'DOWN', 'NEUTRAL']
        
        for i in range(10):
            insert_stmt = text("""
                INSERT INTO predictions 
                (prediction_type, pair, current_price, predicted_price, price_change_percent,
                 predicted_direction, model_name, confidence, confidence_percent, 
                 model_accuracy, r2_score, status, created_at)
                VALUES (:prediction_type, :pair, :current_price, :predicted_price, 
                        :price_change_percent, :predicted_direction, :model_name, 
                        : confidence, :confidence_percent, : model_accuracy, : r2_score, 
                        :status, :created_at)
            """)
            
            current_price = 2030.50 + (i * 2)
            predicted_price = current_price + (5 if i % 2 == 0 else -5)
            
            session.execute(insert_stmt, {
                'prediction_type': 'PRICE',
                'pair': 'XAU/USD',
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': ((predicted_price - current_price) / current_price) * 100,
                'predicted_direction':  directions[i % 3],
                'model_name': 'Ensemble Model',
                'confidence': 0.85 + (i * 0.01),
                'confidence_percent':  87,
                'model_accuracy':  0.87,
                'r2_score': 0.82,
                'status': 'COMPLETED' if i < 5 else 'ACTIVE',
                'created_at':  datetime.now() - timedelta(hours=i)
            })

        logger.info("✅ Predictions seeded")

    def _seed_signals(self, session:  Session):
        """Seed sample trading signals"""
        signal_types = ['BUY', 'SELL', 'HOLD']
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        
        for i in range(15):
            insert_stmt = text("""
                INSERT INTO signals 
                (signal_type, pair, signal_price, strength, strength_percent, confidence, 
                 confidence_percent, risk_level, source, model_name, status, created_at)
                VALUES (:signal_type, :pair, :signal_price, :strength, :strength_percent, 
                        :confidence, :confidence_percent, :risk_level, : source, :model_name, 
                        :status, :created_at)
            """)
            
            session.execute(insert_stmt, {
                'signal_type': signal_types[i % 3],
                'pair': 'XAU/USD',
                'signal_price': 2030.50 + (i * 1. 5),
                'strength': 0.7 + (i * 0.02),
                'strength_percent':  85,
                'confidence': 0.82 + (i * 0.01),
                'confidence_percent':  85,
                'risk_level':  risk_levels[i % 3],
                'source': 'ML_MODEL',
                'model_name':  'LSTM Neural Network',
                'status': 'COMPLETED' if i < 10 else 'TRIGGERED',
                'created_at':  datetime.now() - timedelta(hours=i)
            })

        logger.info("✅ Signals seeded")

    def verify_tables(self) -> bool:
        """
        Verify that all required tables exist
        
        Returns: 
            bool: True if all tables exist
        """
        try:
            logger.info("🔍 Verifying database tables...")

            if not self.engine: 
                logger.error("❌ Engine not initialized")
                return False

            inspector = inspect(self.engine)
            
            required_tables = [
                'users', 'market_data', 'price_history', 'technical_indicators',
                'predictions', 'signals', 'broker_data', 'alerts', 'events'
            ]

            existing_tables = inspector.get_table_names(schema='neydra')
            
            missing_tables = [t for t in required_tables if t not in existing_tables]

            if missing_tables:
                logger.warning(f"⚠️  Missing tables: {', '.join(missing_tables)}")
                return False

            logger. info(f"✅ All required tables exist ({len(required_tables)} tables)")
            
            # Count records in each table
            session = self.session_factory()
            try:
                for table in required_tables:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    logger.info(f"  - {table}: {count} records")
            finally:
                session. close()

            return True

        except Exception as e:
            logger.error(f"❌ Verification failed: {str(e)}")
            return False

    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dict: Database statistics
        """
        try:
            if not self.engine:
                return {}

            session = self.session_factory()
            stats = {}

            try:
                tables = ['users', 'market_data', 'predictions', 'signals', 'broker_data', 'alerts']
                
                for table in tables:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[table] = result.scalar()

                # Get size
                result = session.execute(text("""
                    SELECT pg_size_pretty(pg_database. get_db_size(current_database())) as size
                """))
                stats['database_size'] = result.scalar()

            finally:
                session.close()

            return stats

        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {str(e)}")
            return {}

    def run_all(self, seed_data: bool = True) -> bool:
        """
        Run complete initialization process
        
        Args:
            seed_data: Whether to seed initial data
            
        Returns: 
            bool: True if all steps successful
        """
        logger.info("=" * 80)
        logger.info("🚀 Starting NEYDRA Database Initialization")
        logger.info("=" * 80)

        steps = [
            ("PostgreSQL Connection", self.check_postgresql_connection),
            ("Create Database", self.create_database),
            ("Execute Schema", self.execute_schema),
            ("Create Engine", self.create_engine),
        ]

        if seed_data:
            steps. append(("Seed Initial Data", self.seed_initial_data))

        steps.extend([
            ("Verify Tables", self.verify_tables),
        ])

        success_count = 0
        for step_name, step_func in steps:
            logger.info(f"\n➡️  {step_name}...")
            try:
                if step_func():
                    success_count += 1
                else:
                    logger.error(f"❌ {step_name} failed")
                    return False
            except Exception as e:
                logger.error(f"❌ {step_name} error: {str(e)}")
                return False

        # Display final statistics
        logger.info("\n" + "=" * 80)
        logger.info("📊 Database Statistics:")
        logger.info("=" * 80)
        
        stats = self.get_database_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Database initialization completed successfully!  ({success_count}/{len(steps)} steps)")
        logger.info("=" * 80)

        return True

    def cleanup(self):
        """Cleanup database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("🧹 Database connections cleaned up")


# ===== COMMAND LINE INTERFACE =====
def main():
    """Main entry point for CLI"""
    parser = argparse. ArgumentParser(
        description='NEYDRA Database Initialization Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with defaults
  python init_db.py

  # Initialize with custom host and database
  python init_db.py --host db.example.com --database neydra_prod

  # Initialize without seeding data
  python init_db.py --no-seed

  # Verify only
  python init_db.py --verify-only
        """
    )

    parser.add_argument('--host', default='localhost', help='Database host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432, help='Database port (default: 5432)')
    parser.add_argument('--database', default='neydra', help='Database name (default: neydra)')
    parser.add_argument('--user', default='postgres', help='Database user (default: postgres)')
    parser.add_argument('--password', default='password', help='Database password')
    parser.add_argument('--schema-file', help='Path to schema. sql file')
    parser.add_argument('--no-seed', action='store_true', help='Skip data seeding')
    parser.add_argument('--verify-only', action='store_true', help='Only verify tables')
    parser.add_argument('--stats-only', action='store_true', help='Only show statistics')
    parser.add_argument('--no-create-db', action='store_true', help='Don\'t create database if missing')

    args = parser.parse_args()

    # Create initializer
    initializer = DatabaseInitializer(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        create_db=not args.no_create_db,
        schema_file=args. schema_file
    )

    try:
        if args.verify_only:
            initializer.check_postgresql_connection()
            initializer.create_engine()
            initializer. verify_tables()
        elif args.stats_only:
            initializer.create_engine()
            stats = initializer.get_database_stats()
            print(json.dumps(stats, indent=2))
        else:
            success = initializer.run_all(seed_data=not args.no_seed)
            sys.exit(0 if success else 1)
    finally:
        initializer.cleanup()


if __name__ == '__main__':
    main()