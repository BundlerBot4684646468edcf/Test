# Hotel Decision Simulator

A production-grade SaaS application that helps hotel managers simulate different reception staffing setups, compare AI receptionist options, and optimize their operations for maximum guest satisfaction and profitability.

## Features

- **Interactive Simulator**: Test different staffing configurations in real-time
- **AI Integration**: Compare human vs AI receptionist performance
- **Financial Analysis**: See costs, revenue, and net gain calculations
- **Scenario Management**: Save and compare multiple configurations
- **AI Chat Assistant**: Get personalized advice from an AI assistant
- **User Authentication**: Secure login and registration
- **Responsive Design**: Works on desktop, tablet, and mobile

## Tech Stack

- **Frontend**: Next.js 14 (App Router) + TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **Authentication**: NextAuth.js
- **Database**: Prisma + PostgreSQL
- **Charts**: Recharts
- **AI**: OpenAI API
- **Deployment**: Vercel-ready

## Prerequisites

Before you begin, ensure you have:

- Node.js 18+ installed
- PostgreSQL database (local or cloud)
- OpenAI API key (for chatbot feature)

## Setup Instructions

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd Test

# Install dependencies
npm install
```

### 2. Database Setup

**Option A: Local PostgreSQL**

```bash
# Install PostgreSQL (macOS)
brew install postgresql@15
brew services start postgresql@15

# Create database
createdb hotel_simulator
```

**Option B: Cloud PostgreSQL (Recommended)**

Use [Railway](https://railway.app), [Neon](https://neon.tech), or [Supabase](https://supabase.com):

1. Create a new PostgreSQL database
2. Copy the connection string

### 3. Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/hotel_simulator?schema=public"

# NextAuth
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-secret-key-change-this-in-production"

# OpenAI (for chatbot)
OPENAI_API_KEY="sk-your-openai-api-key"
```

**Generate NEXTAUTH_SECRET:**

```bash
openssl rand -base64 32
```

### 4. Database Migration

```bash
# Generate Prisma Client
npx prisma generate

# Run database migrations
npx prisma db push

# Optional: Open Prisma Studio to view database
npx prisma studio
```

### 5. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
├── app/
│   ├── (app)/                 # Protected routes (requires auth)
│   │   ├── dashboard/         # Dashboard home
│   │   ├── simulator/         # Core simulation tool
│   │   ├── scenarios/         # Saved scenarios
│   │   ├── chat/              # AI chat assistant
│   │   └── settings/          # User settings
│   ├── api/                   # API routes
│   │   ├── auth/              # NextAuth endpoints
│   │   ├── register/          # User registration
│   │   ├── scenarios/         # Scenario CRUD
│   │   └── chat/              # AI chat endpoint
│   ├── auth/                  # Auth pages (login, register)
│   ├── layout.tsx             # Root layout
│   ├── page.tsx               # Landing page
│   └── globals.css            # Global styles
├── components/
│   ├── ui/                    # shadcn/ui components
│   └── sidebar.tsx            # Dashboard sidebar
├── lib/
│   ├── auth.ts                # NextAuth configuration
│   ├── prisma.ts              # Prisma client
│   ├── simulator.ts           # Simulation logic
│   └── utils.ts               # Utility functions
├── prisma/
│   └── schema.prisma          # Database schema
└── types/
    └── next-auth.d.ts         # TypeScript definitions
```

## Usage Guide

### For Hotel Managers

1. **Sign Up**: Create your free account at `/auth/register`
2. **Set Parameters**: Enter your daily request volume and current staffing
3. **Experiment**: Adjust AI costs and capabilities to see the impact
4. **Compare**: Save multiple scenarios and compare results
5. **Get Advice**: Use the AI assistant for personalized recommendations

### Simulation Logic

The simulator calculates:

- **Capacity**: How many requests your team can handle
- **Load Factor**: Demand vs capacity ratio
- **Distribution**: AI vs human handled requests
- **Satisfaction**: Guest satisfaction score (0-100)
- **Financials**: Costs, upsell revenue, and net gain

### Key Metrics Explained

- **Guest Satisfaction**: Based on capacity, load factor, and AI usage
- **Load Factor**: < 1.0 is good (under capacity), > 1.0 means overloaded
- **Net Gain**: Upsell revenue minus total costs (staff + AI)
- **Unhandled Requests**: Requests that exceed total capacity

## Deployment

### Deploy to Vercel

1. Push your code to GitHub
2. Import project in [Vercel](https://vercel.com)
3. Add environment variables in Vercel dashboard
4. Deploy

### Database for Production

Use a managed PostgreSQL service:

- **Railway**: Great for hobbyists, easy setup
- **Neon**: Serverless PostgreSQL, generous free tier
- **Supabase**: Full backend with PostgreSQL
- **AWS RDS**: Enterprise-grade, more complex

## Development

### Adding New Features

1. **Database Changes**: Update `prisma/schema.prisma`, run `npx prisma db push`
2. **New Routes**: Add pages in `app/` directory
3. **New Components**: Add to `components/` directory
4. **API Endpoints**: Add to `app/api/` directory

### Useful Commands

```bash
# Development
npm run dev              # Start dev server
npm run build            # Build for production
npm run start            # Start production server

# Database
npx prisma studio        # Visual database editor
npx prisma generate      # Regenerate Prisma Client
npx prisma db push       # Push schema to database
npx prisma migrate dev   # Create migrations

# Linting
npm run lint             # Run ESLint
```

## Troubleshooting

### Database Connection Issues

```bash
# Test connection
npx prisma db pull

# Reset database (WARNING: deletes all data)
npx prisma db push --force-reset
```

### NextAuth Session Issues

- Ensure `NEXTAUTH_SECRET` is set
- Check `NEXTAUTH_URL` matches your domain
- Clear browser cookies

### OpenAI API Issues

- Verify `OPENAI_API_KEY` is correct
- Check API quota and billing
- Test with a simple request

## Security Notes

- Never commit `.env` files
- Use strong `NEXTAUTH_SECRET` in production
- Enable HTTPS in production
- Regularly update dependencies
- Review Prisma queries for SQL injection risks

## Future Enhancements

- Export scenarios to PDF/Excel
- Multi-hotel support
- Team collaboration features
- Advanced analytics dashboard
- Email notifications
- Mobile app

## License

MIT License - feel free to use for your hotel or modify as needed.

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the code comments
- Open an issue on GitHub

## Credits

Built with Next.js, Prisma, shadcn/ui, and OpenAI.
Designed for hotel managers who want clarity and control.

---

**Made with care for the hospitality industry** 🏨
