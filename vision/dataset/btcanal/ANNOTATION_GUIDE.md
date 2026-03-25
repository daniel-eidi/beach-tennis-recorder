# Guia de Anotacao - Beach Tennis Dataset

## Classes para anotar:
1. **ball** - Bola de beach tennis (pequena, geralmente branca/amarela)
2. **net** - Rede no centro da quadra
3. **court_line** - Linhas da quadra (todas as linhas visiveis)

## Dicas:
- A bola e PEQUENA - faca bounding boxes bem ajustados
- Anote a bola mesmo quando parcialmente oclusa
- A rede deve ter um bbox que cubra toda a extensao visivel
- Linhas da quadra: anote cada segmento visivel separadamente
- Se a bola nao estiver visivel no frame, nao anote (frame sem bola e ok)
- Priorize frames onde a bola esta em jogo (durante rallies)

## Apos anotar:
1. Exporte do Roboflow em formato YOLOv8
2. Extraia em: vision/dataset/btcanal/
3. Rode: python scripts/finetune_roboflow.py --skip-download --dataset-dir dataset/btcanal --epochs 30
