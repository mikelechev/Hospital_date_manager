# Guía Básica de las Aplicaciones

Este repo tiene **tres aplicaciones independientes**. No dependen entre sí — puedes levantar solo la que te interese. Todas se ejecutan desde la raíz del proyecto.

Antes de arrancar cualquiera, activa el entorno virtual:

```bash
source .venv/bin/activate
```

---

## 1. API de Reservas — `src/api_2.py`

**Qué hace:** simula la agenda de un consultorio con 4 huecos fijos (`09:00`, `09:15`, `09:30`, `09:45`). Cuando llega una solicitud de cita, calcula la probabilidad de que el paciente falte y decide:

- Si el hueco está vacío → reserva normal.
- Si el hueco ya tiene 1 paciente → solo permite un **segundo paciente (overbooking)** si el riesgo combinado es bajo:
  - `prob_ambos_vienen < 0.25` **y** `prob_al_menos_uno_venga > 0.8`
- Si el hueco ya tiene 2 pacientes → lo rechaza (saturado).

La probabilidad de ausencia sale del modelo XGBoost (`models/modelo_campeon.json`, ya incluido en el repo) o, si el archivo no existiera, de una fórmula de respaldo basada en edad y días de antelación.

**Cómo correrla:**

```bash
python3 src/api_2.py
```

o con recarga automática:

```bash
./run.sh api
```

Se levanta en `http://localhost:8000`.

**Rutas disponibles:**

| Ruta | Método | Qué hace |
|---|---|---|
| `/` | GET | Página de bienvenida en HTML |
| `/api/estado-agenda` | GET | Devuelve el estado actual de los 4 huecos |
| `/api/evaluar-y-reservar` | POST | Evalúa el riesgo de un paciente y reserva si procede |

**Probarla con curl:**

```bash
curl -X POST http://localhost:8000/api/evaluar-y-reservar \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Paciente Demo",
    "hora": "09:00",
    "age": 35,
    "days_between": 14,
    "ratio_faltas": 0.2,
    "sms_received": 1,
    "weekend": 0
  }'
```

Nota: el estado de la agenda vive solo en memoria — se reinicia cada vez que reinicias la API.

---

## 2. Dashboard de Simulación — `scripts/app.py`

**Qué hace:** simulador Monte Carlo que compara tres estrategias de agendamiento a lo largo de varios días:

1. **Tradicional fijo** — horarios rígidos sin overbooking.
2. **Tradicional flexible** — igual, pero con más margen horario.
3. **IA con overbooking inteligente** — usa el modelo de riesgo para decidir cuándo duplicar un hueco.

Muestra mapas de calor animados (huecos vacíos, asistencia, retrasos, descansos), gráficas de riesgo, tiempos de llegada, sala de espera y ROI, usando los datos reales de `data/dataset_limpio.csv`.

**⚠️ Estado actual: no arranca tal cual.** Necesita `models/modelo_definitivo.joblib`, que **no está en el repo** (solo existe `modelo_campeon.json`, que usa la API, no el dashboard). Para ponerlo a funcionar hace falta entrenar/generar ese `.joblib` o adaptar el script para que reutilice `modelo_campeon.json`.

**Cómo correrlo (una vez resuelto el modelo):**

```bash
streamlit run scripts/app.py
```

Se abre en `http://localhost:8501`.

---

## 3. Chatbot de Admisión — `chatbot/app.py`

**Qué hace:** interfaz conversacional donde un paciente describe su situación en lenguaje natural y un LLM va extrayendo datos estructurados (edad, síntomas, antecedentes, etc.) en tiempo real. Cuando hay suficientes datos, calcula y muestra la probabilidad de ausencia con el mismo modelo XGBoost (`chatbot/models/modelo_campeon.json`).

Funciones principales:

- **Chat libre** (columna izquierda): conversación con el asistente.
- **Panel de estado** (columna derecha): campos ya extraídos, con su nivel de confianza, y los campos que aún faltan.
- **Ficha de historial clínico** (barra lateral): subir o pegar un historial (`.txt`, `.pdf`, `.docx`) para que el LLM extraiga los datos automáticamente, en vez de escribirlos en el chat.
- **Exportar**: descarga CSV de la ficha actual, o la añade a un histórico acumulado (`chatbot/data/historial_pacientes.csv`).
- **Selector de proveedor LLM** (barra lateral): `ollama`, `gemini`, `openai`, `claude` o `custom`. Se puede cambiar en caliente y guardar la API key en `chatbot/.env` desde la propia UI.

**Cómo correrlo:**

```bash
streamlit run chatbot/app.py
```

o:

```bash
./run.sh chatbot
```

Se abre en `http://localhost:8501`.

**Configurar el proveedor de LLM:**

El proveedor por defecto se lee de `chatbot/.env` (variable `DEFAULT_PROVIDER`, por defecto `ollama`). Hoy mismo funciona sin ninguna API key porque **Ollama ya está corriendo localmente** con el modelo `qwen3:8b` descargado.

Para usar otro proveedor, hazlo desde la barra lateral de la app (desplegable "Proveedor" + campo "API Key"), o edita `chatbot/.env`:

```env
DEFAULT_PROVIDER=ollama        # ollama | gemini | openai | claude | custom
DEFAULT_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
GEMINI_API_KEY=...
OPENAI_API_KEY=...
CLAUDE_API_KEY=...             # sk-ant-... , se obtiene en console.anthropic.com
```

`chatbot/.env` está en `.gitignore` — nunca se sube al repositorio.

---

## Resumen rápido

| App | Comando | Puerto | Requiere para funcionar |
|---|---|---|---|
| API de reservas | `python3 src/api_2.py` | 8000 | ✅ Nada extra (modelo ya incluido) |
| Dashboard de simulación | `streamlit run scripts/app.py` | 8501 | ❌ Falta `models/modelo_definitivo.joblib` |
| Chatbot de admisión | `streamlit run chatbot/app.py` | 8501 | ✅ Nada extra (Ollama local ya configurado) |

`run.sh` resume estos mismos comandos: `./run.sh api`, `./run.sh dashboard`, `./run.sh chatbot`.
