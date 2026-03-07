import { NextResponse } from 'next/server'
import { handleApiError } from '@/lib/api-utils';

export async function GET() {
  try {
    const response = await fetch('http://engine:8000/features')
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    return handleApiError(error, 'Failed to fetch features')
  }
}