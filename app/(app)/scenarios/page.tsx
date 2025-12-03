'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { formatCurrency, formatPercent } from '@/lib/utils'
import { Trash2, TrendingUp, TrendingDown, Users, Bot } from 'lucide-react'
import Link from 'next/link'

type Scenario = {
  id: string
  name: string
  description: string | null
  numReceptionists: number
  aiCost: number
  satisfaction: number
  netGain: number
  totalCost: number
  upsellRevenueMonth: number
  createdAt: string
}

export default function ScenariosPage() {
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchScenarios()
  }, [])

  const fetchScenarios = async () => {
    try {
      const response = await fetch('/api/scenarios')
      if (!response.ok) throw new Error('Failed to fetch scenarios')
      const data = await response.json()
      setScenarios(data.scenarios)
    } catch (error) {
      setError('Failed to load scenarios')
    } finally {
      setLoading(false)
    }
  }

  const deleteScenario = async (id: string) => {
    if (!confirm('Are you sure you want to delete this scenario?')) return

    try {
      const response = await fetch(`/api/scenarios/${id}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete')
      setScenarios(scenarios.filter((s) => s.id !== id))
    } catch (error) {
      alert('Failed to delete scenario')
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-gray-600">Loading scenarios...</div>
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Saved Scenarios</h1>
          <p className="mt-2 text-gray-600">
            Review and compare your simulation scenarios
          </p>
        </div>
        <Link href="/simulator">
          <Button>Create New Scenario</Button>
        </Link>
      </div>

      {error && (
        <div className="mb-6 rounded-lg bg-red-50 p-4 text-red-600">
          {error}
        </div>
      )}

      {scenarios.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <div className="text-center">
              <h3 className="text-xl font-semibold text-gray-900">
                No scenarios yet
              </h3>
              <p className="mt-2 text-gray-600">
                Create your first scenario in the simulator
              </p>
              <Link href="/simulator">
                <Button className="mt-4">Go to Simulator</Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {scenarios.map((scenario) => (
            <Card key={scenario.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-xl">{scenario.name}</CardTitle>
                    {scenario.description && (
                      <CardDescription className="mt-1">
                        {scenario.description}
                      </CardDescription>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => deleteScenario(scenario.id)}
                    className="text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
                <div className="mt-3 flex gap-2 text-sm">
                  <div className="flex items-center gap-1 rounded-md bg-blue-100 px-2 py-1 text-blue-700">
                    <Users className="h-3 w-3" />
                    {scenario.numReceptionists} staff
                  </div>
                  <div className="flex items-center gap-1 rounded-md bg-purple-100 px-2 py-1 text-purple-700">
                    <Bot className="h-3 w-3" />
                    {formatCurrency(scenario.aiCost)} AI
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {/* Satisfaction */}
                  <div className="flex items-center justify-between rounded-lg bg-gray-50 p-3">
                    <span className="text-sm text-gray-600">Guest Satisfaction</span>
                    <span className={`font-semibold ${scenario.satisfaction >= 80 ? 'text-green-600' : scenario.satisfaction >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
                      {Math.round(scenario.satisfaction)}%
                    </span>
                  </div>

                  {/* Net Gain */}
                  <div className="flex items-center justify-between rounded-lg bg-gray-50 p-3">
                    <span className="text-sm text-gray-600">Monthly Net Gain</span>
                    <div className="flex items-center gap-1">
                      {scenario.netGain >= 0 ? (
                        <TrendingUp className="h-4 w-4 text-green-600" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-600" />
                      )}
                      <span className={`font-semibold ${scenario.netGain >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatCurrency(scenario.netGain)}
                      </span>
                    </div>
                  </div>

                  {/* Costs */}
                  <div className="flex items-center justify-between rounded-lg bg-gray-50 p-3">
                    <span className="text-sm text-gray-600">Total Costs</span>
                    <span className="font-semibold text-gray-900">
                      {formatCurrency(scenario.totalCost)}
                    </span>
                  </div>

                  {/* Revenue */}
                  <div className="flex items-center justify-between rounded-lg bg-gray-50 p-3">
                    <span className="text-sm text-gray-600">Upsell Revenue</span>
                    <span className="font-semibold text-green-600">
                      {formatCurrency(scenario.upsellRevenueMonth)}
                    </span>
                  </div>

                  {/* Date */}
                  <div className="pt-2 text-xs text-gray-500">
                    Created {new Date(scenario.createdAt).toLocaleDateString()}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
