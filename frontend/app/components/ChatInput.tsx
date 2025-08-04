import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { PaperAirplaneIcon, DocumentTextIcon } from '@heroicons/react/24/outline';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
  ragEnabled?: boolean;
  onRagToggle?: (enabled: boolean) => void;
  agentEnabled?: boolean;
  onAgentToggle?: (enabled: boolean) => void;
  hasUploadedPdf?: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  isLoading, 
  disabled = false, 
  ragEnabled = false, 
  onRagToggle,
  agentEnabled = false,
  onAgentToggle,
  hasUploadedPdf = false 
}) => {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };


  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-gray-700 p-4">
      {(onRagToggle || onAgentToggle) && (
        <div className="flex items-center mb-2 text-sm">
          <span className="text-gray-400 mr-3">Mode:</span>
          <div className="flex space-x-2">
            {/* Chat Mode */}
            <button
              onClick={() => {
                if (onRagToggle) onRagToggle(false);
                if (onAgentToggle) onAgentToggle(false);
              }}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                !ragEnabled && !agentEnabled
                  ? 'bg-neonGreen text-black'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              💬 Chat
            </button>
            
            {/* RAG Mode */}
            {onRagToggle && (
              <button
                onClick={() => {
                  onRagToggle(true);
                  if (onAgentToggle) onAgentToggle(false);
                }}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  ragEnabled && !agentEnabled
                    ? 'bg-neonGreen text-black'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                📚 RAG
              </button>
            )}
            
            {/* Agent Mode */}
            {onAgentToggle && (
              <button
                onClick={() => {
                  onAgentToggle(true);
                  if (onRagToggle) onRagToggle(false);
                }}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  agentEnabled
                    ? 'bg-neonGreen text-black'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                🔬 Agent
              </button>
            )}
          </div>
          
          {/* Mode description */}
          <div className="ml-4 text-xs text-gray-500">
            {agentEnabled ? 'PubMed + Local Docs' : ragEnabled ? 'Local Documents' : 'Standard Chat'}
          </div>
        </div>
      )}
      <div className="flex items-center space-x-2">
        <textarea
          className="flex-1 bg-primary-900 text-white placeholder-gray-400 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-neonGreen resize-none"
          rows={1}
          placeholder="Type a message..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          onClick={handleSend}
          disabled={!message.trim() || disabled}
          className={`p-2 rounded-full ${!message.trim() || disabled ? 'bg-gray-700 text-gray-400' : 'bg-neonGreen text-black'}`}
        >
          <PaperAirplaneIcon className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

export default ChatInput;
