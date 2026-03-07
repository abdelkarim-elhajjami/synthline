import { NextResponse } from 'next/server'
import { handleApiError } from '@/lib/api-utils';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    const response = await fetch('http://engine:8000/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      let errorDetail = 'Generation failed';
      
      try {
        const errorData = JSON.parse(errorText);
        errorDetail = errorData.detail || errorDetail;
      } catch {
        errorDetail = errorText || errorDetail;
      }
      
      return NextResponse.json(
        { error: errorDetail },
        { status: response.status }
      );
    }
    
    // Return success immediately - the actual results will come via WebSocket
    return NextResponse.json({ 
      success: true, 
      message: "Generation running. Updates via WebSocket." 
    });
  } catch (error) {
    return handleApiError(error, 'Failed to generate samples');
  }
}