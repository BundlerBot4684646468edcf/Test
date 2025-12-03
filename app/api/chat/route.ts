// AI Chat Assistant API route

import { NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})

const SYSTEM_PROMPT = `You are a helpful AI assistant for the Hotel Decision Simulator app.
You help hotel managers understand:
- How to optimize their reception staffing
- The benefits of AI receptionists vs human staff
- How to interpret simulation results
- Cost-benefit analysis of different scenarios
- Guest satisfaction factors

Be friendly, clear, and avoid overly technical jargon.
Focus on practical advice for hotel managers aged 40-60 who may not be very technical.`

export async function POST(req: Request) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const { message, history } = await req.json()

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      )
    }

    // Build messages array for OpenAI
    const messages: any[] = [
      { role: 'system', content: SYSTEM_PROMPT },
    ]

    // Add history if provided
    if (history && Array.isArray(history)) {
      messages.push(...history)
    }

    // Add current message
    messages.push({ role: 'user', content: message })

    // Call OpenAI
    const completion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages,
      temperature: 0.7,
      max_tokens: 500,
    })

    const reply = completion.choices[0]?.message?.content || 'Sorry, I could not generate a response.'

    return NextResponse.json({ reply })
  } catch (error: any) {
    console.error('Chat error:', error)

    if (error?.status === 401) {
      return NextResponse.json(
        { error: 'OpenAI API key is invalid or missing' },
        { status: 500 }
      )
    }

    return NextResponse.json(
      { error: 'Failed to process chat message' },
      { status: 500 }
    )
  }
}
