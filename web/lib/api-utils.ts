import { NextResponse } from 'next/server';


type ApiError = Error & { 
  status?: number 
};

export function handleApiError(error: unknown, defaultMessage: string = 'An error occurred') {
  console.error('API error:', error);
  
  let status = 500;
  let message = defaultMessage;
  
  if (error instanceof Error) {
    message = error.message;
    if ('status' in error) {
      status = (error as ApiError).status || 500;
    }
  }
  
  return NextResponse.json({ error: message }, { status });
} 