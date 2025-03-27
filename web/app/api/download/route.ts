import { NextRequest, NextResponse } from 'next/server'
import { handleApiError } from '@/lib/api-utils';

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const path = url.searchParams.get('path');
    
    if (!path) {
      return NextResponse.json({ error: 'Missing path parameter' }, { status: 400 });
    }
    
    // Normalize path by removing 'output/' prefix if present
    const normalizedPath = path.replace(/^output\//, '');
    const apiUrl = `http://engine:8000/files/${normalizedPath}`;
    
    const response = await fetch(apiUrl, {
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });
    
    if (!response.ok) {
      return NextResponse.json(
        { error: `File could not be accessed (${response.status})` }, 
        { status: response.status }
      );
    }
    
    // Extract filename from path with fallback
    const filename = path.split('/').pop() || 'download';
    const fileBuffer = await response.arrayBuffer();
    const contentType = response.headers.get('Content-Type') || 'application/octet-stream';
    
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${filename}"`,
        'Content-Length': fileBuffer.byteLength.toString(),
        'Cache-Control': 'no-cache'
      }
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json({ 
        error: 'Download timed out. The file may be too large.' 
      }, { status: 408 });
    }
    
    return handleApiError(error, 'Failed to download file');
  }
} 