# Hospital Date Manager

Prototipo de optimización de citas hospitalarias: combina un flujo de admisión conversacional con IA, agendamiento tradicional y una lógica de "overbooking" (sobreventa) inteligente basada en el riesgo de que el paciente falte.

El repo tiene **cuatro aplicaciones independientes** (no dependen entre sí, puedes levantar solo la que te interese) más scripts de simulación. Todas se ejecutan desde la raíz del proyecto (`Hospital_date_manager/`).

## 0. Instalación (una sola vez)

Con `run.sh` (recomendado, crea el entorno virtual e instala todo):

```bash
./run.sh setup
```

O manualmente:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r chatbot/requirements.txt
```

A partir de aquí, cada vez que abras una terminal nueva, activa el entorno:

```bash
source .venv/bin/activate
```

---

## 1. API de Reservas — `src/api_2.py`

**Qué hace:** simula la agenda de un consultorio con 4 huecos fijos (`09:00`, `09:15`, `09:30`, `09:45`). Al llegar una solicitud de cita, calcula la probabilidad de que el paciente falte y decide:

- Hueco vacío → reserva normal.
- Hueco con 1 paciente → permite un **segundo paciente (overbooking)** solo si `prob_ambos_vienen < 0.25` **y** `prob_al_menos_uno_venga > 0.8`.
- Hueco con 2 pacientes → rechaza (saturado).

La probabilidad de ausencia sale del modelo XGBoost `models/modelo_campeon.json` (ya incluido en el repo) o, si faltara, de una fórmula de respaldo basada en edad y días de antelación.

**Cómo correrla:**

```bash
python3 src/api_2.py
```

o con recarga automática (uvicorn):

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

> El estado de la agenda vive solo en memoria: se reinicia cada vez que reinicias la API.

---

## 2. Dashboard de Simulación — `scripts/app.py`

**Qué hace:** simulador Monte Carlo que compara tres estrategias de agendamiento a lo largo de varios días (tradicional fijo, tradicional flexible, IA con overbooking inteligente), con mapas de calor animados y gráficas de riesgo/ROI usando `data/dataset_limpio.csv`.

**⚠️ No arranca tal cual todavía.** Necesita `models/modelo_definitivo.joblib`, que **no está en el repo** (solo existe `modelo_campeon.json`, que usa la API, no el dashboard). Para ponerlo a funcionar hace falta entrenar/generar ese `.joblib`, o adaptar el script para que reutilice `modelo_campeon.json`.

**Cómo correrlo (una vez resuelto el modelo):**

```bash
streamlit run scripts/app.py
```

o:

```bash
./run.sh dashboard
```

Se abre en `http://localhost:8501`.

El dashboard carga el `.joblib` al iniciar, así que `catboost` debe estar instalado aunque no se importe directamente (el modelo serializado contiene un componente CatBoost).

---

## 3. Chatbot de Admisión — `chatbot/app.py`

**Qué hace:** interfaz conversacional donde el paciente describe su situación en lenguaje natural y un LLM extrae datos estructurados (edad, síntomas, antecedentes...) en tiempo real. Con suficientes datos, calcula la probabilidad de ausencia con el mismo modelo XGBoost (`chatbot/models/modelo_campeon.json`).

Funciones principales:

- **Chat libre** (columna izquierda).
- **Panel de estado** (columna derecha): campos extraídos, confianza, campos faltantes.
- **Ficha de historial clínico** (barra lateral): subir/pegar historial (`.txt`, `.pdf`, `.docx`) para extracción automática.
- **Exportar**: CSV de la ficha actual o acumulado en `chatbot/exports/`.
- **Selector de proveedor LLM**: `ollama`, `gemini`, `openai`, `claude` o `custom`, configurable en caliente desde la UI.

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

El proveedor por defecto se lee de `chatbot/.env` (crear a partir de `chatbot/.env.example`), variable `DEFAULT_PROVIDER` (por defecto `ollama`). Funciona sin ninguna API key si Ollama ya está corriendo localmente con un modelo descargado (verificado en este equipo: `qwen3:8b`).

```bash
ollama pull qwen3:8b   # si aún no lo tienes
ollama list             # comprobar que está instalado
```

Para usar otro proveedor, hazlo desde la barra lateral de la app (desplegable "Proveedor" + "API Key"), o edita `chatbot/.env`:

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

## 4. Portal Unificado del Paciente — `app_unificado.py`

**Qué hace:** combina el chatbot (`chatbot/app.py`) y el portal de reservas (`scripts/patient.py`) en una sola app de Streamlit con dos pestañas, para que el paciente chatee con el asistente de admisión y reserve un turno sin cambiar de puerto/app.

No duplica lógica: ambos scripts originales fueron refactorizados para exponer `configure_page()` y `render_chatbot_tab()` / `render_agenda_tab()`; `app_unificado.py` configura la página una vez y llama a ambas funciones dentro de `st.tabs(...)`. Los scripts originales siguen funcionando igual por separado.

**Cómo correrlo:**

```bash
streamlit run app_unificado.py
```

o:

```bash
./run.sh portal
```

Se abre en `http://localhost:8501`.

---

## Resumen rápido — todo se ejecuta con `./run.sh`

`run.sh` es el único punto de entrada del proyecto: crea el entorno virtual, instala dependencias y arranca cualquier app o simulación. (El antiguo `chatbot/run.sh`, que creaba su propio entorno virtual duplicado dentro de `chatbot/`, se eliminó — todo pasa ahora por este script.)

| Comando | Qué hace | Puerto | Requiere para funcionar |
|---|---|---|---|
| `./run.sh setup` | Crea `.venv`, instala todas las dependencias y genera `chatbot/.env` si no existe | — | — |
| `./run.sh api` | API FastAPI de reservas (`src/api_2.py`) | 8000 | ✅ Nada extra (modelo ya incluido) |
| `./run.sh dashboard` | Dashboard de simulación (`scripts/app.py`) | 8501 | ❌ Falta `models/modelo_definitivo.joblib` |
| `./run.sh chatbot` | Chatbot de admisión (`chatbot/app.py`) | 8501 | ✅ Nada extra (Ollama local ya configurado) |
| `./run.sh portal` | Portal unificado: chatbot + reservas (`app_unificado.py`) | 8501 | ✅ Igual que el chatbot |
| `./run.sh sim` | Backtest histórico empírico (`scripts/simulacion.py`) | — | ✅ |
| `./run.sh mc` | Monte Carlo, modelo crudo (`scripts/mc.py`) | — | ✅ |
| `./run.sh mc2` | Monte Carlo con calibración isotónica (`scripts/mc_2.py`) | — | ✅ |
| `./run.sh all` | Arranca la API en segundo plano (log en `logs/api.log`) + el portal unificado en primer plano | 8000 + 8501 | ✅ |
| `./run.sh stop` | Detiene la API que quedó corriendo en segundo plano tras `./run.sh all` | — | — |
| `./run.sh help` | Lista todos los comandos disponibles | — | — |

> Nota: Streamlit siempre usa el puerto 8501 por defecto, así que no puedes tener `dashboard`, `chatbot` y `portal` corriendo a la vez sin indicar puertos distintos (`streamlit run <script> --server.port 8502`). La API sí puede convivir con cualquiera de ellas porque usa el puerto 8000 (por eso `./run.sh all` los combina).

---

## Estructura del repositorio

| Ruta | Propósito |
|---|---|
| `app_unificado.py` | Portal unificado: chatbot + reservas en dos pestañas. |
| `scripts/app.py` | Dashboard de simulación y overbooking inteligente. |
| `scripts/patient.py` | Portal de paciente: login TIS, grilla de citas, overbooking asistido por IA. |
| `chatbot/app.py` | UI del chatbot: intake conversacional, historial clínico, exportación. |
| `chatbot/.env.example` | Ejemplo de configuración de Ollama/Gemini. |
| `chatbot/exports/` | Exportaciones de conversaciones e historiales clínicos guardados. |
| `chatbot/conversation/` | Estado de conversación, construcción de prompts, extracción y coordinación con el LLM. |
| `chatbot/providers/` | Adaptadores de proveedor (Ollama, Gemini, OpenAI, Claude). |
| `chatbot/config.py` | Configuración de runtime del chatbot y carga de entorno. |
| `src/api_2.py` | API FastAPI: reserva de huecos, estado de agenda, predicción de riesgo con XGBoost. |
| `scripts/simulacion.py` | Backtest histórico empírico. |
| `scripts/mc.py` | Simulación Monte Carlo con el modelo crudo. |
| `scripts/mc_2.py` | Simulación Monte Carlo con calibración isotónica en memoria. |
| `data/dataset_limpio.csv` | Dataset limpio usado por los scripts de simulación. |
| `models/modelo_campeon.json` | Artefacto XGBoost que carga `src/api_2.py`. |
| `models/modelo_definitivo.joblib` | **Falta en el repo** — lo carga `scripts/app.py` para el dashboard. |
| `models/voting_clf.joblib` | Artefacto de modelo alternativo. |
| `models/calibrated_isotonic_model.joblib` | Artefacto de calibración para evaluación de modelos. |

## Entradas del modelo (XGBoost)

`Age`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `SMS_received`, `Days_between`, `Weekend`, `Ratio_Faltas`, `Gender_M`, `Scheduled_Time_of_Day_Evening`, `Scheduled_Time_of_Day_Morning`.

Columna objetivo: `No-show`.

## Notas

- El chatbot es la interfaz conversacional principal; `src/chatbot.py` es un placeholder vacío de compatibilidad.
- El estado de sesión, agenda y extracción vive en memoria — reiniciar una app resetea el estado.
- `api_key.txt` y `.env` están ignorados por Git y solo deben contener secretos locales.
- Este proyecto es un prototipo, no listo para producción sin validación clínica, persistencia, autenticación y controles de privacidad reales.

## Próximos pasos sugeridos

- Generar o adaptar `models/modelo_definitivo.joblib` para que el dashboard (`scripts/app.py`) arranque.
- Añadir tests automatizados para reglas de reserva, extracción del chat y exportaciones.
- Persistir citas, sesiones e historial clínico en una base de datos.
- Añadir CI para linting, formato y validación en tiempo de ejecución.
- Documentar el entrenamiento del modelo y la preparación del dataset.
