import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { BarChart3, MessageSquare, Save } from 'lucide-react'

export default async function DashboardPage() {
  const session = await getServerSession(authOptions)

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Welcome back, {session?.user?.name}
        </h1>
        <p className="mt-2 text-gray-600">
          Ready to simulate your hotel reception scenarios?
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid gap-6 md:grid-cols-3">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
              <BarChart3 className="h-6 w-6 text-blue-600" />
            </div>
            <CardTitle>Open Simulator</CardTitle>
            <CardDescription>
              Test different staffing and AI setups to find the optimal configuration
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/simulator">
              <Button className="w-full">Go to Simulator</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-green-100">
              <Save className="h-6 w-6 text-green-600" />
            </div>
            <CardTitle>View Saved Scenarios</CardTitle>
            <CardDescription>
              Review and compare your previously saved simulation scenarios
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/scenarios">
              <Button variant="outline" className="w-full">View Scenarios</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100">
              <MessageSquare className="h-6 w-6 text-purple-600" />
            </div>
            <CardTitle>Ask the AI Assistant</CardTitle>
            <CardDescription>
              Get personalized advice and answers to your questions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/chat">
              <Button variant="outline" className="w-full">Open Chat</Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* Getting Started */}
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>
            New to the simulator? Here's how to make the most of it:
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-3">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
              1
            </div>
            <p className="text-sm text-gray-700">
              <strong>Set your baseline:</strong> Enter your current daily request volume and staffing levels
            </p>
          </div>
          <div className="flex gap-3">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
              2
            </div>
            <p className="text-sm text-gray-700">
              <strong>Experiment with AI:</strong> Adjust AI costs and capability to see the impact
            </p>
          </div>
          <div className="flex gap-3">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
              3
            </div>
            <p className="text-sm text-gray-700">
              <strong>Save and compare:</strong> Save multiple scenarios to compare different approaches
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
