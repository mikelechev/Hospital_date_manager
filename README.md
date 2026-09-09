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

Usa `models/modelo_definitivo.joblib`: un `VotingClassifier` (HistGradientBoosting + RandomForest) calibrado con scikit-learn, entrenado sobre `data/dataset_limpio.csv` con 19 variables (edad, comorbilidades, día/mes de cita y de programación, historial de faltas...).

**⚠️ No está versionado en git** (`.gitignore` excluye `*.joblib` por ser un binario pesado). Si clonas el repo desde cero, tienes que generarlo o copiarlo a mano en `models/modelo_definitivo.joblib` antes de arrancar el dashboard o el chatbot; si falta, el dashboard no arranca y el chatbot cae automáticamente a su estimación heurística de respaldo.

**⚠️ Requiere `scikit-learn==1.8.0` exacto.** El `.joblib` es un pickle: con una versión distinta de scikit-learn falla al cargar (`ModuleNotFoundError: No module named '_loss'`, un módulo interno que cambió de sitio entre 1.8 y 1.9). La versión ya viene fijada en `requirements.txt`; si reinstalas dependencias sueltas, respeta ese pin.

**Cómo correrlo:**

```bash
streamlit run scripts/app.py
```

o:

```bash
./run.sh dashboard
```

Se abre en `http://localhost:8501`.

---

## 3. Chatbot de Admisión — `chatbot/app.py`

**Qué hace:** interfaz conversacional donde el paciente describe su situación en lenguaje natural y un LLM extrae datos estructurados (edad, síntomas, antecedentes, historial de faltas...) en tiempo real. Con suficientes datos, calcula la probabilidad de ausencia con el mismo modelo que el dashboard (`models/modelo_definitivo.joblib`) — antes usaba su propia copia de un XGBoost más simple (`chatbot/models/modelo_campeon.json`, ya retirado), con menos variables que este.

Funciones principales:

- **Chat libre** (columna izquierda).
- **Panel de estado** (columna derecha): campos extraídos, confianza, campos faltantes.
- **Ficha de historial clínico** (barra lateral): subir/pegar historial (`.txt`, `.pdf`, `.docx`) para extracción automática.
- **Exportar**: CSV de la ficha actual o acumulado en `chatbot/exports/`.
- **Selector de proveedor LLM**: `ollama`, `gemini`, `openai`, `claude` o `groq`, configurable en caliente desde la UI.

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
DEFAULT_PROVIDER=ollama        # ollama | gemini | openai | claude | groq
DEFAULT_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
GEMINI_API_KEY=...
OPENAI_API_KEY=...             # requiere el paquete 'openai' (ya en chatbot/requirements.txt)
CLAUDE_API_KEY=...             # sk-ant-... , se obtiene en console.anthropic.com
GROQ_API_KEY=...               # gratis en console.groq.com (tier gratuito: ~30 req/min)
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
| `./run.sh dashboard` | Dashboard de simulación (`scripts/app.py`) | 8501 | ❌ Falta `models/modelo_definitivo.joblib` (no va en git, ver sección 2) |
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
| `chatbot/.env.example` | Ejemplo de configuración de los proveedores LLM (Ollama, Gemini, OpenAI, Claude, Groq). |
| `chatbot/exports/` | Exportaciones de conversaciones e historiales clínicos guardados. |
| `chatbot/conversation/` | Estado de conversación, construcción de prompts, extracción y coordinación con el LLM. |
| `chatbot/providers/` | Adaptadores de proveedor (Ollama, Gemini, OpenAI, Claude, Groq). |
| `chatbot/config.py` | Configuración de runtime del chatbot y carga de entorno. |
| `src/api_2.py` | API FastAPI: reserva de huecos, estado de agenda, predicción de riesgo con XGBoost. |
| `scripts/simulacion.py` | Backtest histórico empírico. |
| `scripts/mc.py` | Simulación Monte Carlo con el modelo crudo. |
| `scripts/mc_2.py` | Simulación Monte Carlo con calibración isotónica en memoria. |
| `data/dataset_limpio.csv` | Dataset limpio usado por los scripts de simulación. |
| `models/modelo_campeon.json` | Artefacto XGBoost (13 variables) que carga `src/api_2.py` y los scripts de `scripts/mc*.py`/`simulacion.py`. |
| `models/modelo_definitivo.joblib` | VotingClassifier calibrado (19 variables) que cargan tanto el dashboard (`scripts/app.py`) como el chatbot (`chatbot/prediction/`). **No está en git** (excluido por `.gitignore`, hay que generarlo/copiarlo aparte). |
| `models/voting_clf.joblib` | Artefacto de modelo alternativo. |
| `models/calibrated_isotonic_model.joblib` | Artefacto de calibración para evaluación de modelos. |

## Entradas de los modelos

**`modelo_campeon.json`** (XGBoost — API de reservas y scripts de simulación):

`Age`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `SMS_received`, `Days_between`, `Weekend`, `Ratio_Faltas`, `Gender_M`, `Scheduled_Time_of_Day_Evening`, `Scheduled_Time_of_Day_Morning`.

**`modelo_definitivo.joblib`** (VotingClassifier calibrado — dashboard y chatbot):

`Age`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `SMS_received`, `Days_between`, `Appointment_Day_of_Week`, `Scheduled_Day_of_Week`, `Weekend`, `Appointment_Month`, `Scheduled_Month`, `Faltas_Previas`, `Citas_Previas`, `Ratio_Faltas`, `Gender_M`, `Scheduled_Time_of_Day_Evening`, `Scheduled_Time_of_Day_Morning`.

En el chatbot, las variables de calendario (`Appointment_Day_of_Week`, `Scheduled_Day_of_Week`, `Appointment_Month`, `Scheduled_Month`) se derivan automáticamente de la fecha actual + "días de antelación" que da el paciente, sin preguntarlas por separado.

Columna objetivo en ambos casos: `No-show`.

## Notas

- El chatbot es la interfaz conversacional principal; `src/chatbot.py` es un placeholder vacío de compatibilidad.
- El estado de sesión, agenda y extracción vive en memoria — reiniciar una app resetea el estado.
- `api_key.txt` y `.env` están ignorados por Git y solo deben contener secretos locales.
- Este proyecto es un prototipo, no listo para producción sin validación clínica, persistencia, autenticación y controles de privacidad reales.

## Próximos pasos sugeridos

- Añadir tests automatizados para reglas de reserva, extracción del chat y exportaciones.
- Persistir citas, sesiones e historial clínico en una base de datos.
- Añadir CI para linting, formato y validación en tiempo de ejecución.
- Documentar el entrenamiento del modelo y la preparación del dataset.

## Futuras ideas a implementar

- **Configuración de LLM por sesión, no global**: `LLM_CONFIG` es un singleton a nivel de proceso, así que hoy cambiar de proveedor/modelo afecta a todos los usuarios conectados al mismo tiempo. Moverlo a `st.session_state` permitiría que cada paciente/sesión tenga su propia configuración sin pisar la de los demás.
- **Historial real del paciente en vez de autoreportado**: `Citas_Previas`/`Faltas_Previas` se le preguntan al paciente por chat; si existiera un sistema de pacientes registrados, se podrían recuperar esos conteos de un histórico real en vez de fiarse de lo que recuerda o dice el paciente.
- **Explicabilidad de la predicción**: mostrar al personal qué variables pesaron más en cada predicción (p.ej. con SHAP) en vez de solo el porcentaje de riesgo, para que la decisión de overbooking sea auditable.
- **Reentrenamiento y versionado del modelo**: documentar/automatizar el pipeline de entrenamiento de `modelo_definitivo.joblib`, fijar su versión de scikit-learn junto al artefacto (o exportarlo a un formato más estable como ONNX) para no depender de que el entorno tenga exactamente la misma versión con la que se serializó.
- **Métricas de acierto en producción**: registrar predicción vs. asistencia real para medir el rendimiento del modelo con datos propios del hospital y detectar cuándo hace falta reentrenar.
- **Recordatorios proactivos automáticos**: enviar SMS/email reales a los pacientes de riesgo alto (hoy `SMS_received` es solo un dato de entrada, no una acción) integrando un proveedor de mensajería.
- **Integración con la agenda real (HIS/EHR)** en vez de los 4 huecos simulados de `src/api_2.py`.
- **Ampliar idiomas de la UI**: el sistema de `chatbot/i18n.py` ya está preparado para añadir más idiomas (inglés, catalán...) además de castellano/euskera sin tocar el resto del código.
- **Autenticación y control de acceso** para el personal del hospital, separando el rol de paciente del de administrador/sanitario.
