import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'

export default async function SettingsPage() {
  const session = await getServerSession(authOptions)

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="mt-2 text-gray-600">
          Manage your account settings and preferences
        </p>
      </div>

      <div className="max-w-2xl space-y-6">
        {/* Account Information */}
        <Card>
          <CardHeader>
            <CardTitle>Account Information</CardTitle>
            <CardDescription>
              Your basic account details
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="name">Full Name</Label>
              <Input
                id="name"
                type="text"
                value={session?.user?.name || ''}
                disabled
                className="mt-2"
              />
            </div>
            <div>
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                value={session?.user?.email || ''}
                disabled
                className="mt-2"
              />
            </div>
            <p className="text-sm text-gray-500">
              Account management features coming soon. Contact support to update your details.
            </p>
          </CardContent>
        </Card>

        {/* Simulation Defaults */}
        <Card>
          <CardHeader>
            <CardTitle>Simulation Defaults</CardTitle>
            <CardDescription>
              Default values for the simulator (coming soon)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-500">
              Custom default values for simulation parameters will be available in a future update.
            </p>
          </CardContent>
        </Card>

        {/* Notifications */}
        <Card>
          <CardHeader>
            <CardTitle>Notifications</CardTitle>
            <CardDescription>
              Manage your notification preferences (coming soon)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-500">
              Email notification settings will be available in a future update.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
