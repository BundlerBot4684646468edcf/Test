// Scenarios API routes - GET all and POST new scenario

import { NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { simulate } from '@/lib/simulator'
import { z } from 'zod'

const scenarioSchema = z.object({
  name: z.string().min(1),
  description: z.string().optional(),
  requestsPerDay: z.number().int().min(50).max(500),
  numReceptionists: z.number().int().min(0).max(5),
  salary: z.number().int().min(1500).max(4000),
  aiCost: z.number().int().min(0).max(5000),
  aiCanHandle: z.number().min(0).max(1),
  humanUpsellRate: z.number().min(0).max(0.3),
  aiUpsellRate: z.number().min(0).max(0.3),
  avgUpsellValue: z.number().int().min(5).max(500),
})

// GET all scenarios for current user
export async function GET() {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const scenarios = await prisma.scenario.findMany({
      where: {
        userId: session.user.id,
      },
      orderBy: {
        createdAt: 'desc',
      },
    })

    return NextResponse.json({ scenarios })
  } catch (error) {
    console.error('Get scenarios error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// POST new scenario
export async function POST(req: Request) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const body = await req.json()
    const data = scenarioSchema.parse(body)

    // Run simulation to calculate outputs
    const output = simulate({
      requestsPerDay: data.requestsPerDay,
      numReceptionists: data.numReceptionists,
      salary: data.salary,
      aiCost: data.aiCost,
      aiCanHandle: data.aiCanHandle,
      humanUpsellRate: data.humanUpsellRate,
      aiUpsellRate: data.aiUpsellRate,
      avgUpsellValue: data.avgUpsellValue,
    })

    // Save scenario with inputs and outputs
    const scenario = await prisma.scenario.create({
      data: {
        userId: session.user.id,
        name: data.name,
        description: data.description,
        ...data,
        ...output,
      },
    })

    return NextResponse.json({ scenario }, { status: 201 })
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid request data', details: error.errors },
        { status: 400 }
      )
    }

    console.error('Create scenario error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
