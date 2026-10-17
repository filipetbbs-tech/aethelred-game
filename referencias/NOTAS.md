# Notas das referências

## animações de garras.mp4 (Prototype 2, Metal Gear Rising, Wolverine 2026)

Analisado quadro a quadro a 10 fps.

**Ritmo (Wolverine 14,5–23,5 s)**
- Um golpe a cada 0,3–0,4 s. Preparação de 0,1–0,2 s, golpe em ~1 quadro (0,1 s, com borrão), continuação curta.
- O fim de um golpe já é a preparação do próximo: não volta para a base entre golpes.

**Corpo**
- O tronco faz o trabalho: ombros giram ~80–90°, a coluna dobra quase até a horizontal nos golpes descendentes (15,5 s, 16,0 s, 18,2 s).
- Base larga, um passo para frente a cada golpe; nos golpes grandes, uma perna avança muito e o quadril desce.

**Braços**
- Arcos em diagonal: de trás da cabeça até perto do quadril oposto, braço todo esticado no impacto.
- O braço que não ataca vai para trás e para fora, como contrapeso, com as garras expostas.

**Prototype 2 (2–8 s)**
- Preparação enrolando o tronco: braço bem alto atrás da cabeça, tronco torcido para o lado oposto (2,2 s, 5,1 s).
- Varrida horizontal com o tronco girando 90° e o braço totalmente esticado (3,3 s).
- Postura de combate: curvado, braços à frente e abaixo, garras para frente (3,7–4,9 s).

**Pose de luta (Wolverine 16,5 s, 18,6 s, 20,6 s, 22,7 s)**
- Depois do combo: braços bem abertos, mãos abaixo do quadril, garras para fora e para baixo, levemente agachado; fica nessa pose antes de relaxar.

**O que o pedido destacou**
- Velocidade e fluidez com sensação de impacto; golpes primais; rodopios; corpo inteiro se movendo.
- Pose de luta que dura alguns segundos depois do primeiro ataque.

**Aplicado no jogo (tabela ATT do index.html)**
- Golpes 1–4: ~0,7 s cada, preparação ~0,27 s, golpe + continuação ~0,12 s; encadeados ficam a ~0,4 s um do outro.
- Golpe 2 com rodopio de 360° (spin) acertando em volta; golpe 5 é o salto girando do Mixamo.
- Guarda (GUARD) mantida por 3 s depois de um golpe; andar ou pular sai dela.

## Coesão do combo (combo B)

- Problema do combo A: o golpe final do Mixamo é lento e gracioso, de espadachim, e começa de uma postura ereta neutra,
  quebrando o fluxo dos golpes curvados e rápidos; e os golpes 1-4 tinham todos o mesmo ritmo, sem crescendo.
- Combo B: escalada de ritmo e peso (0,55 s → 0,55 → 0,7 → 0,85 → 1,2 s; dano 1, 1, 1, 2, 4), tudo no mesmo estilo,
  e o fim de cada golpe deixa o tronco torcido para o lado do braço seguinte (continuidade de yaw entre golpes).
- Final: bote com as duas garras, com a preparação longa (agacha e segura) antes de um golpe rápido, para dar peso.
