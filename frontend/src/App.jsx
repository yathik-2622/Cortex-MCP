import React, { useState, useRef, useEffect } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Bot, User, Terminal, ChevronDown, ChevronRight, Loader2, AlertCircle } from 'lucide-react';
import './App.css'; // Importing your CSS file

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002/api/chat';

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am the Nexus Agent. I have access to Web Search, Wikipedia, Weather, and a Knowledge Base. How can I help you today?',
      logs: []
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setInput('');
    setError(null);
    
    // Add user message and a placeholder for the assistant
    setMessages(prev => [
      ...prev, 
      userMessage, 
      { role: 'assistant', content: '', logs: [] }
    ]);
    setIsLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content })
      });

      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);

      // --- STREAMING LOGIC ---
      // We use a Reader to process the response line-by-line
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last partial line in the buffer
        buffer = lines.pop() || ''; 

        for (const line of lines) {
          if (!line.trim()) continue;
          
          try {
            const data = JSON.parse(line);
            
            setMessages(prev => {
              const newMessages = [...prev];
              const lastMsg = newMessages[newMessages.length - 1];

              // 1. Handle "log" type (Tools)
              if (data.type === 'log') {
                lastMsg.logs = [...(lastMsg.logs || []), data];
              } 
              // 2. Handle "token" type (Real-time Typing)
              else if (data.type === 'token') {
                if (!lastMsg.content) lastMsg.content = '';
                lastMsg.content += data.content;
              }
              // 3. Handle "answer" type (Full block fallback)
              else if (data.type === 'answer') {
                lastMsg.content = data.content; 
              } 
              // 4. Handle errors
              else if (data.type === 'error') {
                setError(data.message);
              }

              return newMessages;
            });
          } catch (e) {
            console.error('Error parsing JSON chunk:', e);
          }
        }
      }

    } catch (err) {
      setError(err.message);
      setMessages(prev => {
        const newMessages = [...prev];
        // Remove empty assistant bubble if failed completely
        if (!newMessages[newMessages.length - 1].content && newMessages[newMessages.length - 1].logs.length === 0) {
          newMessages.pop();
        }
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="logo-icon">
          <Bot size={24} color="white" />
        </div>
        <div>
          <h1>Nexus Agent</h1>
          <p className="subtitle">Powered by Groq & MCP</p>
        </div>
      </header>

      {/* Chat Area */}
      <main className="chat-area">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-row ${msg.role}`}>
            
            {/* Assistant Avatar */}
            {msg.role === 'assistant' && (
              <div className="avatar ai">
                <Bot size={20} color="white" />
              </div>
            )}

            <div className="message-content">
              {/* Message Bubble */}
              <div className={`bubble ${msg.role}`}>
                {msg.role === 'assistant' ? (
                  <div className="markdown-content">
                    {msg.content ? (
                      <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
                    ) : (
                      /* Loading Indicator inside bubble */
                      (msg.logs.length === 0 && isLoading) ? (
                        <div className="loading-indicator">
                          <Loader2 size={16} className="spinner" />
                          <span>Thinking...</span>
                        </div>
                      ) : (
                        <span style={{color: '#94a3b8', fontStyle: 'italic'}}>Generative step...</span>
                      )
                    )}
                  </div>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>

              {/* Logs / Thoughts Section (Collapsible) */}
              {msg.role === 'assistant' && msg.logs && msg.logs.length > 0 && (
                <LogViewer logs={msg.logs} isStreaming={idx === messages.length - 1 && isLoading} />
              )}
            </div>

            {/* User Avatar */}
            {msg.role === 'user' && (
              <div className="avatar user">
                <User size={20} color="white" />
              </div>
            )}
          </div>
        ))}
        
        {error && (
          <div className="error-banner">
            <AlertCircle size={16} style={{display:'inline', marginRight:'8px'}} />
            {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* Input Area */}
      <div className="input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask complex questions..."
            className="text-input"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? <Loader2 size={20} className="spinner" /> : <Send size={20} />}
          </button>
        </form>
      </div>
    </div>
  );
}

// Sub-component to show the "Thinking" logs
function LogViewer({ logs, isStreaming }) {
  const [isOpen, setIsOpen] = useState(false);

  // Auto-expand logs only when they first start streaming
  useEffect(() => {
    if (isStreaming && logs.length > 0) setIsOpen(true);
  }, [isStreaming]);

  if (!logs || logs.length === 0) return null;

  return (
    <div className="logs-container">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="toggle-logs-btn"
      >
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {isOpen ? 'Hide Thoughts' : `View Thoughts (${logs.length})`}
      </button>
      
      {isOpen && (
        <div className="logs-box">
          {logs.map((log, i) => (
            <div key={i} className="log-entry">
              <Terminal size={14} style={{marginTop: '2px', flexShrink: 0}} />
              <div>
                <span className="log-tool">[{log.tool || 'System'}]:</span>{' '}
                <span>{log.message}</span>
                {log.input && (
                  <div className="log-input">
                    Input: {truncate(log.input, 80)}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isStreaming && (
            <div style={{paddingLeft: '22px', color: '#4f46e5', fontStyle: 'italic'}}>
              ... processing
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function truncate(str, n) {
  return (str.length > n) ? str.substr(0, n - 1) + '...' : str;
}

export default App;