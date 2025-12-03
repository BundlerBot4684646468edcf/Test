import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { BarChart3, Users, TrendingUp } from 'lucide-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      {/* Navigation */}
      <nav className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="text-xl font-bold text-blue-600">
              Hotel Decision Simulator
            </div>
            <div className="flex gap-4">
              <Link href="/auth/login">
                <Button variant="ghost">Log in</Button>
              </Link>
              <Link href="/auth/register">
                <Button>Sign up free</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 text-center">
        <div className="mx-auto max-w-3xl">
          <h1 className="mb-6 text-5xl font-bold leading-tight text-gray-900">
            Simulate your reception before you hire.
          </h1>
          <p className="mb-8 text-xl text-gray-600">
            Test different staffing + AI setups and see cost, upsell and guest
            satisfaction before you make the decision.
          </p>
          <div className="flex justify-center gap-4">
            <Link href="/auth/register">
              <Button size="lg" className="text-lg">
                Sign up free
              </Button>
            </Link>
            <Link href="/auth/login">
              <Button size="lg" variant="outline" className="text-lg">
                Log in
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="container mx-auto px-4 py-16">
        <h2 className="mb-12 text-center text-3xl font-bold text-gray-900">
          How it works
        </h2>
        <div className="grid gap-8 md:grid-cols-3">
          <Card>
            <CardHeader>
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
                <Users className="h-6 w-6 text-blue-600" />
              </div>
              <CardTitle>1. Set your parameters</CardTitle>
              <CardDescription>
                Enter your daily request volume, current staffing, and AI assistant costs
              </CardDescription>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
                <BarChart3 className="h-6 w-6 text-blue-600" />
              </div>
              <CardTitle>2. See real-time results</CardTitle>
              <CardDescription>
                View capacity, guest satisfaction, upsell revenue, and net gain instantly
              </CardDescription>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
                <TrendingUp className="h-6 w-6 text-blue-600" />
              </div>
              <CardTitle>3. Make confident decisions</CardTitle>
              <CardDescription>
                Save scenarios, compare options, and choose the best setup for your hotel
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </section>

      {/* Features */}
      <section className="container mx-auto px-4 py-16">
        <div className="rounded-2xl bg-blue-600 px-8 py-16 text-center text-white">
          <h2 className="mb-4 text-3xl font-bold">
            Built for hotel managers who want clarity
          </h2>
          <p className="mb-8 text-lg text-blue-100">
            No technical jargon. Just clear insights to help you optimize your
            reception operations.
          </p>
          <Link href="/auth/register">
            <Button size="lg" variant="secondary" className="text-lg">
              Get started now
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-gray-50 py-8">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p>&copy; 2024 Hotel Decision Simulator. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}
