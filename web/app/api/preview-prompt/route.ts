import { NextResponse } from 'next/server'
import { handleApiError } from '@/lib/api-utils';

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const response = await fetch('http://engine:8000/preview-prompt', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json(
        { error: errorData.detail || 'Failed to generate prompt preview' },
        { status: response.status }
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    return handleApiError(error, 'Failed to generate prompt preview')
  }
} 