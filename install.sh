#!/bin/bash
# MCP Memory - Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/TWFBusiness/mpc-memoria/main/install.sh | bash

set -e

MCP_DIR="$HOME/.mcp-memoria"
DATA_DIR="$MCP_DIR/data"

echo "=== Installing MCP Memory ==="

# Create directories
mkdir -p "$MCP_DIR"
mkdir -p "$DATA_DIR"

# Download server.py
echo "Downloading server..."
curl -fsSL "https://raw.githubusercontent.com/TWFBusiness/mpc-memoria/main/server.py" -o "$MCP_DIR/server.py"

# Create venv
echo "Creating virtual environment..."
python3 -m venv "$MCP_DIR/.venv"

# Install dependencies
echo "Installing dependencies..."
"$MCP_DIR/.venv/bin/pip" install --upgrade pip -q
"$MCP_DIR/.venv/bin/pip" install mcp -q

# Ask about embeddings
read -p "Install embeddings for semantic search? (+150MB RAM) [y/N]: " INSTALL_EMB
EMBEDDING_ENABLED="false"
if [[ "$INSTALL_EMB" =~ ^[Yy]$ ]]; then
    echo "Installing sentence-transformers..."
    "$MCP_DIR/.venv/bin/pip" install sentence-transformers numpy -q
    EMBEDDING_ENABLED="true"
fi

echo ""
echo "=== Installation complete! ==="
echo ""

# Auto-configure Claude Code if available
if command -v claude &> /dev/null; then
    read -p "Configure Claude Code automatically? [Y/n]: " CONFIGURE_CLAUDE
    if [[ ! "$CONFIGURE_CLAUDE" =~ ^[Nn]$ ]]; then
        echo "Adding MCP to Claude Code (global scope)..."
        claude mcp add memory "$MCP_DIR/.venv/bin/python" "$MCP_DIR/server.py" --scope user -e MCP_MEMORY_EMBEDDING="$EMBEDDING_ENABLED" 2>/dev/null && \
            echo "✓ Claude Code configured! Restart Claude Code to activate." || \
            echo "⚠ Could not configure automatically. See manual instructions below."
    fi
fi

echo ""
echo "=== Configuration ==="
echo ""
echo "Claude Code (run this command):"
echo "  claude mcp add memory $MCP_DIR/.venv/bin/python $MCP_DIR/server.py --scope user -e MCP_MEMORY_EMBEDDING=$EMBEDDING_ENABLED"
echo ""
echo "Cursor (~/.cursor/mcp.json):"
echo "{"
echo "  \"mcpServers\": {"
echo "    \"memory\": {"
echo "      \"command\": \"$MCP_DIR/.venv/bin/python\","
echo "      \"args\": [\"$MCP_DIR/server.py\"],"
echo "      \"env\": { \"MCP_MEMORY_EMBEDDING\": \"$EMBEDDING_ENABLED\" }"
echo "    }"
echo "  }"
echo "}"
echo ""
echo "=== Available tools ==="
echo "  - memory_context: Auto-recall relevant context"
echo "  - memory_search: Search memories"
echo "  - memory_save: Save decisions/patterns"
echo "  - memory_list: List recent memories"
echo "  - memory_stats: Show statistics"
echo "  - memory_delete: Remove memory"
echo ""
