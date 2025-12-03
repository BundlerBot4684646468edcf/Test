'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { simulate, SimulationInput, SimulationOutput } from '@/lib/simulator'
import { formatCurrency, formatPercent } from '@/lib/utils'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'

export default function SimulatorPage() {
  const router = useRouter()
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [saveSuccess, setSaveSuccess] = useState(false)

  // Input state
  const [inputs, setInputs] = useState<SimulationInput>({
    requestsPerDay: 250,
    numReceptionists: 2,
    salary: 2300,
    aiCost: 1500,
    aiCanHandle: 0.6,
    humanUpsellRate: 0.07,
    aiUpsellRate: 0.04,
    avgUpsellValue: 35,
  })

  // Calculate outputs in real-time
  const outputs: SimulationOutput = simulate(inputs)

  // Update input helper
  const updateInput = (field: keyof SimulationInput, value: number) => {
    setInputs((prev) => ({ ...prev, [field]: value }))
    setSaveSuccess(false)
    setSaveError('')
  }

  // Save scenario
  const handleSave = async () => {
    const name = prompt('Enter a name for this scenario:')
    if (!name) return

    setSaving(true)
    setSaveError('')
    setSaveSuccess(false)

    try {
      const response = await fetch('/api/scenarios', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          description: `${inputs.numReceptionists} receptionists + AI (€${inputs.aiCost}/mo)`,
          ...inputs,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to save scenario')
      }

      setSaveSuccess(true)
      setTimeout(() => {
        router.push('/scenarios')
      }, 1000)
    } catch (error) {
      setSaveError('Failed to save scenario. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  // Chart data
  const capacityData = [
    { name: 'Human', value: outputs.humanCapacity },
    { name: 'AI', value: outputs.aiCapacity },
  ]

  const handledData = [
    { name: 'Human Handled', value: Math.round(outputs.humanHandled) },
    { name: 'AI Handled', value: Math.round(outputs.aiHandled) },
    { name: 'Unhandled', value: Math.round(outputs.unhandled) },
  ]

  const COLORS = ['#3b82f6', '#10b981', '#ef4444']

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Reception Simulator</h1>
        <p className="mt-2 text-gray-600">
          Adjust parameters to see real-time impact on capacity, satisfaction, and costs
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* INPUTS */}
        <div className="space-y-6">
          {/* Demand */}
          <Card>
            <CardHeader>
              <CardTitle>Demand</CardTitle>
              <CardDescription>
                Total requests per day across all channels (calls, emails, WhatsApp, front desk)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Requests per day</Label>
                  <span className="text-sm font-semibold">{inputs.requestsPerDay}</span>
                </div>
                <Slider
                  value={[inputs.requestsPerDay]}
                  onValueChange={([value]) => updateInput('requestsPerDay', value)}
                  min={50}
                  max={500}
                  step={10}
                />
              </div>
            </CardContent>
          </Card>

          {/* Human Reception */}
          <Card>
            <CardHeader>
              <CardTitle>Human Reception</CardTitle>
              <CardDescription>
                Your human receptionist team size and costs
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Number of receptionists</Label>
                  <span className="text-sm font-semibold">{inputs.numReceptionists}</span>
                </div>
                <Slider
                  value={[inputs.numReceptionists]}
                  onValueChange={([value]) => updateInput('numReceptionists', value)}
                  min={0}
                  max={5}
                  step={1}
                />
              </div>
              <div>
                <Label htmlFor="salary">Salary per receptionist (€/month)</Label>
                <Input
                  id="salary"
                  type="number"
                  value={inputs.salary}
                  onChange={(e) => updateInput('salary', Number(e.target.value))}
                  min={1500}
                  max={4000}
                  step={100}
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>

          {/* AI Assistant */}
          <Card>
            <CardHeader>
              <CardTitle>AI Assistant</CardTitle>
              <CardDescription>
                AI receptionist costs and capability level
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="aiCost">AI monthly cost (€/month)</Label>
                <Input
                  id="aiCost"
                  type="number"
                  value={inputs.aiCost}
                  onChange={(e) => updateInput('aiCost', Number(e.target.value))}
                  min={0}
                  max={5000}
                  step={100}
                  className="mt-2"
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>AI can fully handle (% of simple requests)</Label>
                  <span className="text-sm font-semibold">{Math.round(inputs.aiCanHandle * 100)}%</span>
                </div>
                <Slider
                  value={[inputs.aiCanHandle * 100]}
                  onValueChange={([value]) => updateInput('aiCanHandle', value / 100)}
                  min={0}
                  max={100}
                  step={5}
                />
              </div>
            </CardContent>
          </Card>

          {/* Upselling */}
          <Card>
            <CardHeader>
              <CardTitle>Upselling</CardTitle>
              <CardDescription>
                Revenue generation through upselling services
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Human upsell rate (%)</Label>
                  <span className="text-sm font-semibold">{Math.round(inputs.humanUpsellRate * 100)}%</span>
                </div>
                <Slider
                  value={[inputs.humanUpsellRate * 100]}
                  onValueChange={([value]) => updateInput('humanUpsellRate', value / 100)}
                  min={0}
                  max={30}
                  step={1}
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>AI upsell rate (%)</Label>
                  <span className="text-sm font-semibold">{Math.round(inputs.aiUpsellRate * 100)}%</span>
                </div>
                <Slider
                  value={[inputs.aiUpsellRate * 100]}
                  onValueChange={([value]) => updateInput('aiUpsellRate', value / 100)}
                  min={0}
                  max={30}
                  step={1}
                />
              </div>
              <div>
                <Label htmlFor="avgUpsellValue">Average upsell value (€)</Label>
                <Input
                  id="avgUpsellValue"
                  type="number"
                  value={inputs.avgUpsellValue}
                  onChange={(e) => updateInput('avgUpsellValue', Number(e.target.value))}
                  min={5}
                  max={500}
                  step={5}
                  className="mt-2"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* OUTPUTS */}
        <div className="space-y-6">
          {/* Key Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Key Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-gray-600">Guest Satisfaction</div>
                  <div className={`mt-1 text-2xl font-bold ${outputs.satisfaction >= 80 ? 'text-green-600' : outputs.satisfaction >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
                    {Math.round(outputs.satisfaction)}%
                  </div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-gray-600">Load Factor</div>
                  <div className={`mt-1 text-2xl font-bold ${outputs.loadFactor <= 1 ? 'text-green-600' : 'text-red-600'}`}>
                    {outputs.loadFactor.toFixed(2)}
                  </div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-gray-600">Monthly Net Gain</div>
                  <div className={`mt-1 text-2xl font-bold ${outputs.netGain >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatCurrency(outputs.netGain)}
                  </div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="text-sm text-gray-600">Total Costs</div>
                  <div className="mt-1 text-2xl font-bold text-gray-900">
                    {formatCurrency(outputs.totalCost)}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Request Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Request Distribution</CardTitle>
              <CardDescription>
                How requests are handled across your team
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={handledData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {handledData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Handled by humans:</span>
                  <span className="font-semibold">{Math.round(outputs.humanHandled)} / day</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Handled by AI:</span>
                  <span className="font-semibold">{Math.round(outputs.aiHandled)} / day</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Unhandled:</span>
                  <span className="font-semibold text-red-600">{Math.round(outputs.unhandled)} / day</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Financial Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Financial Summary (Monthly)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-600">Staff Cost:</span>
                  <span className="font-semibold">{formatCurrency(outputs.staffCost)}</span>
                </div>
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-600">AI Cost:</span>
                  <span className="font-semibold">{formatCurrency(inputs.aiCost)}</span>
                </div>
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-600">Total Costs:</span>
                  <span className="font-semibold">{formatCurrency(outputs.totalCost)}</span>
                </div>
                <div className="flex justify-between border-b pb-2">
                  <span className="text-gray-600">Upsell Revenue:</span>
                  <span className="font-semibold text-green-600">+{formatCurrency(outputs.upsellRevenueMonth)}</span>
                </div>
                <div className="flex justify-between pt-2">
                  <span className="text-lg font-semibold">Net Gain:</span>
                  <span className={`text-lg font-bold ${outputs.netGain >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatCurrency(outputs.netGain)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Save Button */}
          <Card>
            <CardContent className="pt-6">
              <Button
                onClick={handleSave}
                disabled={saving}
                className="w-full"
                size="lg"
              >
                {saving ? 'Saving...' : 'Save This Scenario'}
              </Button>
              {saveSuccess && (
                <div className="mt-3 rounded-md bg-green-50 p-3 text-sm text-green-600">
                  Scenario saved! Redirecting...
                </div>
              )}
              {saveError && (
                <div className="mt-3 rounded-md bg-red-50 p-3 text-sm text-red-600">
                  {saveError}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
