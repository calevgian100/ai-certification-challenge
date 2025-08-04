import { NextRequest, NextResponse } from 'next/server';

// Get API URL from environment variable or use default
// Use explicit IP address instead of localhost to avoid IPv6 issues
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Extract required parameters for agent
    const { query, thread_id, user_profile } = body;

    // Validate required parameters
    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' },
        { status: 400 }
      );
    }

    // Forward request to backend agent API
    console.log(`Sending agent request to: ${API_URL}/agents/query`);
    const response = await fetch(`${API_URL}/agents/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        thread_id: thread_id || `user_${Date.now()}`,
        user_profile,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Backend agent API error: ${response.status} - ${errorText}`);
      return NextResponse.json(
        { error: `Backend agent API error: ${response.status}` },
        { status: response.status }
      );
    }

    // Get the JSON response from the backend
    const agentResponse = await response.json();
    
    // Return the agent response
    return NextResponse.json(agentResponse);

  } catch (error) {
    console.error('Agent API route error:', error);
    return NextResponse.json(
      { error: 'Internal server error in agent API route' },
      { status: 500 }
    );
  }
}
