# MCP Memoria

Memória persistente para Claude Code com busca semântica e duas camadas de contexto.

## Instalação Rápida

```bash
curl -fsSL https://raw.githubusercontent.com/TWFBusiness/mpc-memoria/main/install.sh | bash
```

Ou manualmente:

```bash
git clone https://github.com/TWFBusiness/mpc-memoria.git ~/.claude/mcp-memoria
cd ~/.claude/mcp-memoria
./install.sh
```

Reinicie o Claude Code após instalar.

## Como Funciona

### Duas Camadas de Memória

| Camada | Local | Uso |
|--------|-------|-----|
| **Global** | `~/.claude/memoria/` | Padrões pessoais, preferências que valem para todos projetos |
| **Projeto** | `.claude/memoria/` | Decisões específicas do projeto atual |

### Busca Inteligente

- **FTS5**: Busca textual instantânea (sempre ativo)
- **Embeddings**: Busca semântica em background (opcional, +150MB RAM)

Com embeddings, buscas como "como configurei autenticação" encontram memórias sobre "JWT com refresh token" mesmo sem palavras em comum.

### Indexação em Background

Embeddings são processados de forma assíncrona:
1. Você salva uma memória → resposta imediata (SQLite)
2. Worker em background gera embedding
3. Próximas buscas já incluem o novo conteúdo

Não há bloqueio ou lentidão ao salvar.

## Uso

### Salvar Memórias

```
"salve que eu prefiro pytest ao invés de unittest"
"lembre que neste projeto usamos PostgreSQL com Tortoise ORM"
"guarde globalmente: sempre usar Black para formatação"
```

### Buscar Memórias

```
"o que já decidimos sobre testes?"
"como configuramos o banco de dados?"
"quais são meus padrões de código?"
```

### Comandos Diretos (opcional)

```
memoria_salvar(
  conteudo="FastAPI sempre 100% async, jamais sync",
  tipo="padrao",
  escopo="global",
  tags="python,fastapi,async"
)

memoria_buscar(query="autenticação", escopo="ambos")

memoria_listar(tipo="decisao", escopo="projeto", limite=10)

memoria_deletar(id="abc123", escopo="global")
```

## Tipos de Memória

| Tipo | Quando Usar |
|------|-------------|
| `decisao` | Escolhas técnicas, trade-offs, soluções de bugs |
| `padrao` | Preferências de código, libs favoritas, estilo |
| `arquitetura` | Estrutura do projeto, fluxos, integrações |
| `preferencia` | Configurações pessoais gerais |
| `todo` | Tarefas pendentes |
| `nota` | Anotações diversas |

## Configuração

Arquivo: `~/.claude/.mcp.json`

```json
{
  "mcpServers": {
    "memoria": {
      "command": "~/.claude/mcp-memoria/.venv/bin/python",
      "args": ["~/.claude/mcp-memoria/server.py"],
      "env": {
        "MEMORIA_EMBEDDING": "true"
      }
    }
  }
}
```

### Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `MEMORIA_EMBEDDING` | `true` | Habilita busca semântica |
| `MEMORIA_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Modelo de embedding |

### Modelos de Embedding

| Modelo | RAM | Qualidade | Idiomas |
|--------|-----|-----------|---------|
| `all-MiniLM-L6-v2` | ~80MB | Boa | EN (ok para código) |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~150MB | Boa | Multi (melhor PT-BR) |
| `all-mpnet-base-v2` | ~400MB | Excelente | EN |

## Backup e Restauração

### Exportar

```bash
# Memórias
cp ~/.claude/memoria/global.db ~/backup/memoria-global.db

# Preferências
cp ~/.claude/CLAUDE.md ~/backup/CLAUDE.md

# Memórias de projeto específico
cp /path/to/projeto/.claude/memoria/projeto.db ~/backup/projeto-x.db
```

### Importar

```bash
# Memórias
cp ~/backup/memoria-global.db ~/.claude/memoria/global.db

# Preferências
cp ~/backup/CLAUDE.md ~/.claude/CLAUDE.md
```

## Estrutura de Arquivos

```
~/.claude/
├── .mcp.json              # Configuração do MCP server
├── CLAUDE.md              # Instruções globais (opcional)
├── memoria/
│   └── global.db          # SQLite - memórias globais
└── mcp-memoria/
    ├── server.py          # Servidor MCP
    ├── install.sh         # Instalador
    ├── requirements.txt   # Dependências Python
    └── .venv/             # Ambiente virtual

~/seu-projeto/
└── .claude/
    └── memoria/
        └── projeto.db     # SQLite - memórias do projeto
```

## Requisitos

- Python 3.10+
- Claude Code CLI
- ~10MB RAM (FTS only) ou ~150MB RAM (com embeddings)

## Desinstalação

```bash
rm -rf ~/.claude/mcp-memoria
rm -rf ~/.claude/memoria
rm ~/.claude/.mcp.json
```

## Licença

MIT
