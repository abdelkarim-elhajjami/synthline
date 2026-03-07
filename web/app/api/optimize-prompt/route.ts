import { NextResponse } from 'next/server'
import { handleApiError } from '@/lib/api-utils';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    const response = await fetch('http://engine:8000/optimize-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      let errorDetail = 'Prompt optimization failed';
      
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
      message: "Optimization running. Updates via WebSocket." 
    });
  } catch (error) {
    return handleApiError(error, 'Failed to optimize prompt');
  }
} 