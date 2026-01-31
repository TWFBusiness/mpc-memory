#!/bin/bash
# MCP Memoria - Instalador
# Uso: curl -fsSL https://raw.githubusercontent.com/TWFBusiness/mpc-memoria/main/install.sh | bash

set -e

MCP_DIR="$HOME/.claude/mcp-memoria"
CONFIG_FILE="$HOME/.claude/.mcp.json"

echo "=== Instalando MCP Memoria ==="

# Criar diretório
mkdir -p "$MCP_DIR"
mkdir -p "$HOME/.claude/memoria"

# Baixar server.py (ou copiar se local)
if [ -f "$(dirname "$0")/server.py" ]; then
    cp "$(dirname "$0")/server.py" "$MCP_DIR/server.py"
else
    curl -fsSL "https://raw.githubusercontent.com/TWFBusiness/mpc-memoria/main/server.py" -o "$MCP_DIR/server.py"
fi

# Criar venv
echo "Criando ambiente virtual..."
python3 -m venv "$MCP_DIR/.venv"

# Instalar dependências
echo "Instalando dependências..."
"$MCP_DIR/.venv/bin/pip" install --upgrade pip -q
"$MCP_DIR/.venv/bin/pip" install mcp -q

# Perguntar sobre embeddings
read -p "Instalar embeddings para busca semântica? (+150MB RAM) [s/N]: " INSTALL_EMB
EMBEDDING_ENABLED="false"
if [[ "$INSTALL_EMB" =~ ^[Ss]$ ]]; then
    echo "Instalando sentence-transformers..."
    "$MCP_DIR/.venv/bin/pip" install sentence-transformers numpy -q
    EMBEDDING_ENABLED="true"
fi

# Configurar MCP
echo "Configurando Claude Code..."
cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "memoria": {
      "command": "$MCP_DIR/.venv/bin/python",
      "args": ["$MCP_DIR/server.py"],
      "env": {
        "MEMORIA_EMBEDDING": "$EMBEDDING_ENABLED"
      }
    }
  }
}
EOF

echo ""
echo "=== Instalação completa! ==="
echo ""
echo "Reinicie o Claude Code para ativar."
echo ""
echo "Ferramentas disponíveis:"
echo "  - memoria_buscar: busca memórias"
echo "  - memoria_salvar: salva decisões/padrões"
echo "  - memoria_listar: lista memórias"
echo ""
