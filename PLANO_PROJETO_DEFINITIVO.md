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

## Retomando numa sessão local (app desktop, aba Code)

1. Baixar o projeto: `git clone https://github.com/filipetbbs-tech/aethelred-game.git` (ex.: em `C:\Jogos`).
2. Git só com a conta pessoal **dentro da pasta do jogo** (o GitHub Desktop do notebook usa a conta de trabalho):
   - `git config user.name` / `git config user.email` locais com o nome e e-mail pessoais;
   - remoto com o usuário pessoal, para o Windows guardar um login separado:
     `git remote set-url origin https://filipetbbs-tech@github.com/filipetbbs-tech/aethelred-game.git`;
   - no primeiro push, o login no navegador deve ser o do GitHub **pessoal**;
   - não adicionar este projeto no GitHub Desktop.
3. Copiar as vozes para `sons/` (e `sons/criatura/`): elas não estão no GitHub.
4. Blender (linha de comando) para modelos; Unity 6 HDRP quando começar o projeto definitivo.

## Estado atual do protótipo (navegador)

- Link de teste (artifact): https://claude.ai/artifact/LeXCghigcRKv5ZvWT3Gpqw — publicado a partir de `index.html`.
  Os modelos/animações vão como arquivos `.b64.txt` (o `loadBinary` cai no `.b64.txt` se o binário falhar).
- Perfil padrão: Mixamo (`aethelred_personagem.glb`). Braço monstruoso padrão na transformação: o **antigo**
  (`aethelred_braco_mutante_antigo.glb`).
- Combo de 5 golpes (perfil Mixamo): velocidades 1.8 / 1.45 / 1.45 / 1.4 / 1.5. O 5º é cópia do golpe do combo
  original para não mudar o modelo original.
- Inimigos: só vikings (`modelos/viking_rig_armas_rigidas.glb`), com cerco, revezamento de ataque, agarrão/arremesso,
  reação a golpe que cancela o ataque, ataques mais lentos (`VK.ritmoGolpe`), recuperação após dano, andam quando perto.
- Celular: controle na tela no layout do DualSense (analógico, △ transformar, ○ esquivar, ✕ pular, □ atacar,
  R1 agarrar, R2 correr, pausa, ⛶ tela cheia). A coluna de botões de opções está escondida; as opções ficam no teclado
  (I inimigos, P pausa, O som, M música, B câmera lenta, N mapa, G invencível, L versão da esquiva, T pose T).

## Modelos novos (testes com Meshy) — estado

- **Modelo A** (montado: corpo + cabeça + cabelo, ver tabela de scripts acima): melhor geometria (rosto detalhado,
  tranças com enfeites). Recebeu o rosto da arte conceito projetado e o cabelo acobreado (`warpA.py`, `projectA.py`,
  `finishA.py`). Problema: o **formato** do rosto é diferente da arte.
- **Modelo B** (gerado inteiro pelo Meshy): formato do rosto mais próximo da arte, menos detalhe, cabelo em blocos.
  Recebeu a mesma projeção (`warp.py`, `project.py`, `hairmask.py`, `haircolor.py`, `fw_bake.py`).
- Arte conceito: prancha com vistas de frente, 3/4, perfil e costas (imagem enviada na conversa; pedir de novo ao
  usuário, de preferência cada vista separada em alta resolução).
- Próximo passo combinado: tentar **deformar a malha do rosto do modelo A** para as proporções da arte (pontos de
  frente e de perfil, deformação suave) e projetar de novo; se não ficar bom, gerar no Meshy uma cabeça nova a partir
  da frente + perfil da prancha.
- Os arquivos `.glb` montados ficaram na sessão da nuvem; para refazer, usar os scripts com os arquivos do Meshy.
