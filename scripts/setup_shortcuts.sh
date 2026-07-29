#!/bin/bash
# Setup keyboard shortcuts for PyStreamAI

add_shortcuts() {
  if [ -f ~/.zshrc ]; then
    RC_FILE=~/.zshrc
  elif [ -f ~/.bashrc ]; then
    RC_FILE=~/.bashrc
  else
    echo "❌ No shell config found"; return 1
  fi
  
  if grep -q "dash-pystreamai" "$RC_FILE"; then
    echo "⚠️  Shortcuts already installed"; return 0
  fi
  
  cat >> "$RC_FILE" << 'ALIASES'

# PyStreamAI dashboard shortcuts
alias dash-pystreamai='pystreamai dashboard --static'
alias dash-pystreamai-live='pystreamai dashboard'
alias dash-pystreamai-export='pystreamai dashboard --export /tmp/pystreamai_metrics.json && echo ✓ Exported'
ALIASES
  
  echo "✅ Shortcuts added to $RC_FILE"
  echo "   Run: source $RC_FILE"
}

remove_shortcuts() {
  sed -i '' '/# PyStreamAI dashboard shortcuts/,/alias dash-pystreamai-export=/d' ~/.zshrc 2>/dev/null
  sed -i '' '/# PyStreamAI dashboard shortcuts/,/alias dash-pystreamai-export=/d' ~/.bashrc 2>/dev/null
  echo "✅ Shortcuts removed"
}

case "${1:-}" in --remove) remove_shortcuts ;; *) add_shortcuts ;; esac
