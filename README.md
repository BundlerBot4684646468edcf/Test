# Stempl — Digital Loyalty Card System

A modern loyalty card platform for small businesses (cafés, bars, salons) in South Tyrol. Customers get real passes in Apple Wallet and Google Wallet (no app download required). When staff scan a QR code at checkout, the stamp count updates live on the pass.

## Tech Stack

- **Backend**: Node.js + TypeScript, Express
- **Database**: PostgreSQL (Prisma ORM)
- **Frontend Admin**: Next.js (React), Tailwind CSS
- **Deployment**: Docker

## Quick Start

### Prerequisites

- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 14+ (via Docker)

### 1. Install dependencies

```bash
npm install
```

### 2. Setup environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start database

```bash
docker-compose up -d
```

### 4. Run migrations

```bash
npm run prisma:migrate
```

### 5. Start development server

```bash
npm run dev
```

Server runs on `http://localhost:3000`

## Project Structure

```
stempl-loyalty-wallet/
├── src/
│   ├── index.ts           # Express server entry point
│   ├── config.ts          # Configuration management
│   ├── routes/            # API endpoints
│   ├── services/          # Business logic
│   ├── wallets/           # Apple Wallet & Google Wallet integrations
│   └── utils/             # Helpers & utilities
├── prisma/
│   └── schema.prisma      # Database schema
├── docker-compose.yml     # PostgreSQL & Adminer
└── package.json
```

## Database Models

- **Business**: Loyalty program details (name, logo, stamp goal, reward text)
- **Customer**: Enrolled customers (wallet serial, current stamps)
- **StampEvent**: Individual stamp records (timestamp, staff member)
- **WalletDevice**: Apple & Google Wallet registration data

## Important: Certificates & API Keys

⚠️ **NEVER commit these to git:**

- Apple Pass Type ID Certificate (.p12)
- Apple WWDR Certificate
- Apple APNs Authentication Key (.p8)
- Google Service Account JSON

They go in `/certs` (see `.gitignore`). Get setup instructions from the next phase of development.

## Development Commands

```bash
npm run dev              # Start with hot reload
npm run build           # Build TypeScript
npm run start           # Run compiled code
npm run prisma:studio   # Open Prisma Studio (visual DB editor)
npm run typecheck       # Run TypeScript type checking
npm run lint            # Run ESLint
```

## Next Steps

Phase 1: ✓ **Prisma Schema + Backend Groundwork** — Complete
- Prisma schema created
- Express server scaffolded
- Docker database configured
- Environment variables documented

Phase 2 (Next): **Google Wallet Integration**
- Setup Google Cloud Project
- Create Service Account
- Implement loyaltyClass/loyaltyObject endpoints
- Test with real Google Wallet
