#!/bin/bash
# Setup MCP Memoria

set -e

cd "$(dirname "$0")"

echo "=== MCP Memoria Setup ==="

# Criar venv
if [ ! -d ".venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv .venv
fi

# Ativar e instalar deps
echo "Instalando dependências..."
source .venv/bin/activate
pip install --upgrade pip
pip install mcp

echo ""
echo "=== Setup completo! ==="
echo ""
echo "Para habilitar embeddings (opcional, +150MB RAM):"
echo "  source .venv/bin/activate"
echo "  pip install sentence-transformers numpy"
echo "  # E mude MEMORIA_EMBEDDING para true no ~/.claude/.mcp.json"
echo ""
echo "Reinicie o Claude Code para carregar o MCP server."
