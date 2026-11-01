# Plano do projeto definitivo — Aethelred

O jogo no navegador (Three.js, `index.html`) é um **protótipo**: serve para validar combate,
tempos de golpe, IA dos inimigos e números de balanceamento. A versão definitiva será feita
numa engine. Este arquivo guarda o que foi decidido para retomar quando os testes de
modelos terminarem.

## Engine escolhida: Unity 6 LTS com HDRP

Motivos:
- Roda confortável no notebook atual (16 GB de RAM). A Unreal pede 32 GB para trabalhar bem.
- Tudo é programado em C# (texto), o que permite escrever praticamente o jogo inteiro em
  conjunto com o Claude, inclusive scripts que automatizam o editor.
- O HDRP tem o necessário para o visual da arte conceito: pele com subsurface scattering
  (diffusion profile), shader de olhos (córnea/refração), shader de cabelo, iluminação HDRI,
  tone mapping, oclusão ambiente. Referência: demo "Enemies" e pacote Digital Human da Unity.

A **Unreal Engine 5** continua como opção futura (MetaHuman, Nanite, Groom), se a máquina
ganhar 32 GB de RAM. Modelos e animações preparados servem nas duas.

### Máquina
- Notebook com Intel i9-13900H, RTX 4060 Laptop (8 GB VRAM), 16 GB DDR5, SSD NVMe.
- Unity ocupa ~10 GB instalada + alguns GB por projeto.
- Forçar o Unity a usar a GPU NVIDIA (Configurações → Sistema → Tela → Elementos gráficos →
  Unity → Alto desempenho), trabalhar na tomada e com o NitroSense em modo desempenho.

### Passos de instalação (quando for começar)
1. Instalar o Unity Hub e o Unity 6 LTS, com o módulo de build para Windows.
2. Criar o projeto com o template **3D (HDRP)**.
3. Sessão local do Claude Code na pasta do projeto, usando **a conta pessoal**
   (separada da conta de trabalho: pasta de configuração própria via `CLAUDE_CONFIG_DIR`
   e git configurado só com a conta pessoal do GitHub nesse repositório).
4. Primeiro teste: "sala de retrato" com a personagem, pele, olhos e iluminação de estúdio,
   comparando lado a lado com a arte conceito.

### Diferenças de pipeline na Unity (em relação à Unreal)
- **Rosto com expressões:** malha base com os 52 blendshapes do padrão ARKit, ajustada
  (wrap) ao rosto do Meshy, com bake dos detalhes. Captura de atuação pelo app
  *Unity Face Capture* (iPhone, padrão ARKit).
- **Polígonos:** sem Nanite. Cada modelo precisa de redução (decimate) + bake de normal map e
  cor, e de níveis de detalhe (LODs). Gameplay ~30–60 mil triângulos; cutscene ~80–150 mil.
- **Cabelo:** faixas com transparência (hair cards); fio a fio só com pacote experimental.

## Personagem principal (testes com Meshy)

Montagem feita no Blender (via `bpy`), scripts em `ferramentas/personagem_meshy/`:

| Script | O que faz |
|---|---|
| `combine2.py` | Junta corpo + cabeça (busto) + cabelo. Escala da cabeça calculada pelo pescoço (S=0.23), corte do busto inclinado (CUT=-0.67, TILT=0.5), cabelo com escala própria (KX=1.45, KY=1.65, KZ=1.45, RY=0.09, RZ=0.06). |
| `seam.py` | Remove o toco de pescoço do corpo e "cola" a borda do pescoço da cabeça na pele do corpo. |
| `tone.py` | Iguala os tons de pele da cabeça e do corpo (meio-termo entre os dois). |
| `scalp.py` | Afunda o couro cabeludo só onde o cabelo cobre, para não atravessar o cabelo. |
| `skinmask.py` | Marca no corpo a área de roupa (top/calcinha) e a densidade de sardas por região. |
| `skin.py` | Sardas e poros procedurais no corpo, fora da roupa (FS=320, FS2=640, FD=2.2, FD2=1.8, FR=0.5, FSTR=1.0, cor 0.52/0.28/0.18, PD=0.8). |
| `bake.py` | Grava as sardas/poros na textura de cor e no normal map do corpo (4096). |
| `rlib.py` | Iluminação e câmera para os renders de conferência. |

Os caminhos dos arquivos de entrada nos scripts apontam para os uploads da sessão em que
foram usados; ajustar ao reaproveitar.

Próximos passos pendentes do personagem:
- Testar o modelo do Meshy já gerado com cabeça + cabelo + corpo juntos e comparar a qualidade.
- Aproximar o rosto da arte conceito: gerar no Meshy a partir de várias vistas da prancha e
  projetar a pintura da arte na textura do rosto no Blender.
- Redução com bake para o jogo, esqueleto (Mixamo) e, para cutscenes, rig facial.
