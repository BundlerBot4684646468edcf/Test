import dotenv from 'dotenv';

dotenv.config();

export const config = {
  // Server
  port: parseInt(process.env.PORT || '3000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  appUrl: process.env.APP_URL || 'http://localhost:3000',
  dashboardUrl: process.env.DASHBOARD_URL || 'http://localhost:3001',

  // Database
  databaseUrl: process.env.DATABASE_URL,

  // Apple Wallet (PassKit)
  apple: {
    passTypeId: process.env.APPLE_PASS_TYPE_ID || 'pass.com.stempl.loyalty',
    teamId: process.env.APPLE_TEAM_ID,
    certPath: process.env.APPLE_CERT_PATH,
    certPassword: process.env.APPLE_CERT_PASSWORD,
    wwdrCertPath: process.env.APPLE_WWDR_CERT_PATH,
  },

  // Apple Push Notification Service
  apns: {
    keyId: process.env.APPLE_APNS_KEY_ID,
    teamId: process.env.APPLE_APNS_TEAM_ID,
    authKeyPath: process.env.APPLE_APNS_AUTH_KEY_PATH,
  },

  // Google Wallet API
  google: {
    serviceAccountPath: process.env.GOOGLE_WALLET_SERVICE_ACCOUNT_PATH,
    issuerId: process.env.GOOGLE_WALLET_ISSUER_ID,
  },

  // Email (future)
  smtp: {
    host: process.env.SMTP_HOST,
    port: parseInt(process.env.SMTP_PORT || '587', 10),
    user: process.env.SMTP_USER,
    password: process.env.SMTP_PASSWORD,
  },
};

// Validate required config
if (!config.databaseUrl) {
  throw new Error('DATABASE_URL environment variable is required');
}
