# Suporte a Múltiplos Modelos (Multi-Model)

O backend YOLO permite que um único control tag use **vários modelos ao mesmo tempo**.
Isso é útil quando você tem modelos especializados em classes diferentes — por exemplo,
um modelo que detecta placas de veículos e outro que detecta contêineres — e quer as
predições de todos eles em um único resultado enviado ao Label Studio.

O ponto mais importante: **os modelos não são carregados juntos na GPU**. Cada modelo é
processado de vez e, ao final, é descarregado da GPU antes do próximo ser executado.
Assim, dois modelos grandes não precisam caber na memória da GPU simultaneamente.

---

## Como usar

Separe os nomes dos arquivos por vírgula no atributo `model_path` do control tag:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image"
                  model_score_threshold="0.25"
                  model_path="plate_id_1.pt,container_v12.pt">
    <Label value="container-face"/>
    <Label value="truck-plate"/>
  </RectangleLabels>
</View>
```

Espaços em branco ao redor da vírgula são ignorados, então
`model_path="plate_id_1.pt, container_v12.pt"` também funciona.

Os arquivos precisam estar no diretório de modelos (`/app/models` no Docker, ou o
diretório configurado em `MODEL_ROOT`).

### Exemplo com os modelos do projeto

Os dois modelos abaixo têm as mesmas classes (`container-face`, `container-front`,
`container-id`, `truck-front`, `truck-plate`), mas foram treinados separadamente.
Usando os dois juntos, as detecções de ambos aparecem no mesmo resultado:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_score_threshold="0.25"
                  model_path="plate_id_1.pt,container_v12.pt">
    <Label value="container-face" background="green"/>
    <Label value="container-front" background="blue"/>
    <Label value="container-id"    background="red"/>
    <Label value="truck-front"     background="yellow"/>
    <Label value="truck-plate"     background="purple"/>
  </RectangleLabels>
</View>
```

O resultado retorna as regiões dos dois modelos combinadas em um único `result`,
com um `score` médio calculado sobre todas as regiões.

---

## Funciona com todos os control tags

O recurso é implementado na classe base `ControlModel`, então vale para qualquer
control tag que use modelo:

| Control tag      | Tarefa                  | Exemplo                                              |
|------------------|-------------------------|------------------------------------------------------|
| `RectangleLabels` | Detecção de objetos     | `model_path="model_a.pt,model_b.pt"`                 |
| `PolygonLabels`   | Segmentação (polígonos) | `model_path="seg_a.pt,seg_b.pt"`                     |
| `BrushLabels`     | Segmentação (máscara)   | `model_path="seg_a.pt,seg_b.pt"`                     |
| `Choices`         | Classificação           | `model_path="cls_a.pt,cls_b.pt"`                     |
| `KeyPointLabels`  | Pose / keypoints        | `model_path="pose_a.pt,pose_b.pt"`                   |
| `VideoRectangle`  | Tracking em vídeo       | `model_path="det_a.pt,det_b.pt"`                     |

---

## Mapeamento de classes (label map)

Cada modelo mantém o **seu próprio** mapeamento de classes, porque modelos diferentes
podem ter nomes de classes diferentes (ou até conjuntos diferentes de classes).

Use o atributo `predicted_values` nos `<Label>` para alinhar as classes de cada modelo
aos rótulos do Label Studio:

```xml
<RectangleLabels name="label" toName="image" model_score_threshold="0.25"
                model_path="plate_id_1.pt,container_v12.pt">
  <!-- Classes do modelo de placas -->
  <Label value="Placa" predicted_values="truck-plate,plate"/>
  <!-- Classes do modelo de contêineres -->
  <Label value="Container" predicted_values="container-face,container-front"/>
</RectangleLabels>
```

O control tag só é aceito se pelo menos um dos modelos tiver classes compatíveis com
os `<Label>` da configuração. Se nenhum modelo tiver classes compatíveis, o backend
registra um erro no log listando as classes disponíveis de cada modelo.

---

## Como funciona internamente

### Processamento sequencial

O método `predict_regions_for_tasks()` (em `control_models/base.py`) é responsável por
executar os modelos. Para um control tag com múltiplos modelos:

1. O primeiro modelo é carregado (ou obtido do cache) e executado sobre **todas** as
   tarefas do batch.
2. O modelo é descarregado da GPU (`offload_model()`).
3. O próximo modelo é carregado e executado sobre todas as tarefas.
4. As regiões de todos os modelos são somadas no mesmo resultado.

Como o loop externo é por control tag (e não por task), um batch com N tarefas e M
modelos carrega cada modelo **uma única vez**, e não `N x M` vezes.

### Descarregamento da GPU

A cada modelo processado, o backend:

1. Descarta o `predictor` do Ultralytics — é essa cópia que fica na GPU.
2. Move o modelo de volta para a CPU (os pesos ficam em cache na memória RAM).
3. Chama `torch.cuda.empty_cache()` para liberar a memória da GPU.

Na predição seguinte, o modelo é reutilizado a partir do cache de CPU, sem precisar
recarregar o arquivo `.pt` do disco.

> **Atenção:** o cache de CPU mantém os modelos na RAM. Se os modelos forem muito
> grandes e o conjunto for grande, a RAM também pode ser um limite.

### Modelos default não são mais pré-carregados

Antes, o backend carregava 6 modelos padrão no startup, mesmo que a configuração
usasse apenas modelos customizados. Agora **só os modelos referenciados pela
configuração são carregados** — o que reduz o uso de memória e acelera o start.

### Single-model inalterado

Se o `model_path` contiver apenas um modelo, o comportamento é idêntico ao anterior:
nada é descarregado e o modelo fica fixo na GPU para reuso.

---

## Parâmetros

| Parâmetro               | Tipo   | Default | Descrição                                                                 |
|-------------------------|--------|---------|---------------------------------------------------------------------------|
| `model_path`            | string | Nenhum  | Caminho do modelo ou lista separada por vírgula com vários modelos.       |
| `model_score_threshold` | float  | 0.5     | Confiança mínima para manter uma detecção.                                |

A variável de ambiente `ALLOW_CUSTOM_MODEL_PATH` precisa estar habilitada (o padrão
do `docker-compose.yml` é `true`).

---

## Solução de problemas

**O control tag não aparece / "No label map built"**
Nenhum dos modelos tem classes compatíveis com os `<Label>`. Confira no log as
classes disponíveis de cada modelo e use `predicted_values`.

**A GPU fica sem memória mesmo com dois modelos**
Verifique se os dois modelos estão realmente sendo processados em sequência (log de
`load`/`predict`/`offload`). O cache de CPU mantém os pesos na RAM; se o problema
for RAM, reduza o tamanho dos modelos ou use menos modelos por control tag.

**Só um dos modelos está retornando detecções**
Provavelmente o outro modelo não tem classes mapeadas no `<Label>`, ou o
`model_score_threshold` está alto demais para ele.

---

## Testes

Os testes da funcionalidade estão em `tests/test_multi_model.py`:

- Parsing do `model_path` com vírgulas, espaços e entradas vazias.
- Mesclagem de regiões de vários modelos.
- Ordem de execução `load → predict → offload` (garante que só um modelo fica na GPU).
- Single-model não faz offload.
- End-to-end via `/predict` mesclando detecções de dois modelos.
- End-to-end com os modelos reais `plate_id_1.pt` + `container_v12.pt`.

Rode com:

```bash
pip install -r requirements-test.txt
pytest tests/test_multi_model.py -v
```
