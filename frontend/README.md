<div align="center">
  <img src="../backend/docs/logo.png" alt="AGENT-K" width="300">
  <h1>AGENT-K Mission Console</h1>
  <h3>Multi-Agent Kaggle GrandMaster (🧐)</h3>
  <p><em>Real-time dashboard for monitoring autonomous Kaggle competition missions</em></p>
</div>

<div align="center">
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-16-black.svg" alt="Next.js 16"></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-19-blue.svg" alt="React 19"></a>
  <a href="https://tailwindcss.com/"><img src="https://img.shields.io/badge/Tailwind-4-38bdf8.svg" alt="Tailwind CSS 4"></a>
</div>

---

## Overview

The AGENT-K Mission Console is a Next.js dashboard for real-time monitoring of autonomous Kaggle competition missions. It provides visualization of the multi-agent orchestration, evolution progress, tool calls, and mission state through the AG-UI protocol.

---

## Features

| Feature | Description |
|---------|-------------|
| **Mission Dashboard** | Real-time overview of mission phases, progress, and status |
| **Evolution Visualization** | Fitness charts tracking population improvement across generations |
| **Tool Call Inspection** | Detailed view of agent tool calls with thinking blocks and results |
| **Phase Cards** | Status cards for each mission phase (Discovery, Research, Prototype, Evolution, Submission) |
| **Memory View** | Inspection of agent memory entries and checkpoints |
| **Research View** | Display of leaderboard analysis and research findings |
| **Logs View** | Filterable log stream of mission events |

---

## Installation

### Prerequisites

- Node.js 20+
- pnpm

### Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Run database migrations
pnpm db:migrate

# Start development server
pnpm dev
```

Your app should now be running on [localhost:3000](http://localhost:3000).

---

## Key Components

Located in `components/agent-k/`:

### Mission Dashboard

```
mission-dashboard.tsx    # Main dashboard container
├── phase-card.tsx       # Phase status cards
├── task-card.tsx        # Individual task cards
└── tool-call-card.tsx   # Tool call details with thinking
```

### Evolution View

```
evolution-view.tsx       # Evolution phase visualization
├── fitness-chart.tsx    # Fitness progression chart (Recharts)
└── [leaderboard submissions, convergence status]
```

### Research View

```
research-view.tsx        # Research findings display
├── leaderboard analysis
├── paper citations
└── strategy recommendations
```

### Memory View

```
memory-view.tsx          # Memory entries and checkpoints
├── session/persistent/global scopes
└── checkpoint restoration
```

### Logs View

```
logs-view.tsx            # Filterable event log
├── task status filters
└── tool type filters
```

### Plan View

```
plan-view.tsx            # Mission plan overview
├── phase objectives
└── task dependencies
```

---

## State Management

### AG-UI Protocol

The frontend communicates with the backend through the AG-UI protocol, receiving real-time updates via Server-Sent Events (SSE).

### Mission State Hook

```typescript
import { useAgentKState } from '@/hooks/use-agent-k-state';

function MissionDashboard() {
  const { mission, ui, dispatch } = useAgentKState();
  
  return (
    <div>
      <p>Phase: {mission.currentPhase}</p>
      <p>Progress: {mission.overallProgress}%</p>
    </div>
  );
}
```

### Type Definitions

All TypeScript types are defined in `lib/types/agent-k.ts`:

```typescript
// Core types
type MissionPhase = 'discovery' | 'research' | 'prototype' | 'evolution' | 'submission';
type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'blocked' | 'skipped';

// Tool types
type ToolType = 'web_search' | 'kaggle_mcp' | 'code_executor' | 'memory' | 'browser';

// State interfaces
interface MissionState { ... }
interface EvolutionState { ... }
interface ResearchFindings { ... }
```

---

## Project Structure

```
frontend/
├── app/                        # Next.js app router
│   ├── (auth)/                 # Authentication routes
│   │   ├── login/              # Login page
│   │   └── register/           # Registration page
│   ├── (chat)/                 # Main chat/dashboard routes
│   │   ├── chat/               # Chat interface
│   │   ├── api/                # API routes
│   │   └── page.tsx            # Home page
│   ├── globals.css             # Global styles
│   └── layout.tsx              # Root layout
│
├── components/
│   ├── agent-k/                # AGENT-K specific components
│   │   ├── mission-dashboard.tsx
│   │   ├── phase-card.tsx
│   │   ├── task-card.tsx
│   │   ├── tool-call-card.tsx
│   │   ├── evolution-view.tsx
│   │   ├── fitness-chart.tsx
│   │   ├── research-view.tsx
│   │   ├── memory-view.tsx
│   │   ├── logs-view.tsx
│   │   └── plan-view.tsx
│   ├── ui/                     # Shared UI components (shadcn/ui)
│   └── elements/               # Custom elements
│
├── hooks/
│   ├── use-agent-k-state.tsx   # Mission state management
│   ├── use-artifact.ts         # Artifact handling
│   └── use-auto-resume.ts      # Mission resumption
│
├── lib/
│   ├── types/
│   │   ├── agent-k.ts          # AGENT-K TypeScript types
│   │   └── events.ts           # Event types
│   ├── ai/                     # AI SDK integration
│   ├── db/                     # Database (Drizzle ORM)
│   └── utils/                  # Utility functions
│
├── public/                     # Static assets
│
└── tests/                      # Playwright tests
    ├── e2e/                    # End-to-end tests
    └── pages/                  # Page objects
```

---

## Running Locally

### Development

```bash
pnpm dev          # Start with Turbopack
```

### Production Build

```bash
pnpm build        # Build for production
pnpm start        # Start production server
```

### Testing

```bash
pnpm test         # Run Playwright tests
```

### Linting

```bash
pnpm lint         # Check code style
pnpm format       # Fix code style
```

---

## Configuration

### Environment Variables

Create a `.env.local` file with:

```bash
# Database
DATABASE_URL="postgresql://..."

# Authentication
AUTH_SECRET="your-auth-secret"

# AI Gateway (for Vercel deployments)
AI_GATEWAY_API_KEY="your-gateway-key"

# Backend Connection
AGENT_K_BACKEND_URL="http://localhost:8000"
```

### Database

The frontend uses PostgreSQL with Drizzle ORM:

```bash
pnpm db:generate  # Generate migrations
pnpm db:migrate   # Apply migrations
pnpm db:studio    # Open Drizzle Studio
```

---

## Connecting to Backend

### Development Proxy

The frontend proxies API requests to the backend. Configure in `proxy.ts`:

```typescript
const BACKEND_URL = process.env.AGENT_K_BACKEND_URL || 'http://localhost:8000';
```

### AG-UI Event Stream

The frontend subscribes to the backend's SSE endpoint for real-time updates:

```typescript
const eventSource = new EventSource('/api/agent-k/events');

eventSource.onmessage = (event) => {
  const patch = JSON.parse(event.data);
  dispatch({ type: 'APPLY_PATCH', patch });
};
```

---

## UI Components

Built with:

- **[shadcn/ui](https://ui.shadcn.com/)** - Accessible component primitives
- **[Radix UI](https://radix-ui.com/)** - Unstyled, accessible components
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first styling
- **[Recharts](https://recharts.org/)** - Composable charting library
- **[Framer Motion](https://www.framer.com/motion/)** - Animation library
- **[Lucide React](https://lucide.dev/)** - Icon library

---

## License

MIT License - see [LICENSE](LICENSE) for details.
