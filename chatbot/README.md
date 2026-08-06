# 🏥 Hospital Conversational Chatbot

Este proyecto desarrolla un chatbot conversacional basado en Inteligencia Artificial cuyo objetivo es mantener una conversación natural con un paciente mientras recopila automáticamente información para predecir la probabilidad de No-Show[cite: 4].

## 🏗️ Arquitectura

El proyecto sigue estrictamente los principios de Clean Architecture y SOLID.

- **Capa de Interfaz:** Streamlit (`app.py`), sin lógica de negocio[cite: 1].
- **Capa de Conversación:** Gestión de estado (`PatientState`) y coordinación del flujo conversacional[cite: 2].
- **Capa de Proveedores:** Interfaces desacopladas para Ollama y Gemini (`BaseProvider`)[cite: 2].
- **Capa de Predicción:** Aislamiento del modelo XGBoost (`Predictor`)[cite: 2].

El chatbot **NO** agenda citas, **NO** diagnostica y **NO** sustituye al personal sanitario[cite: 4].

## 🚀 Instalación y Ejecución (Linux/macOS)

1. Clona el repositorio.
2. Copia la plantilla de variables de entorno:
   ```bash
   cp .env.example .env
   ```
3. Crea y activa un entorno virtual (recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Ejecutar la interfaz (Streamlit)

Arranca la UI de Streamlit desde la carpeta `chatbot`:

```bash
streamlit run app.py
```

La interfaz incluye un panel lateral que muestra el estado del paciente y, en la sección "Export Conversation", un control para crear un export de la conversación actual.

## 💾 Exportar conversaciones

- Formatos soportados: **JSON** y **CSV**.
- Ubicación del export: por defecto `chatbot/exports/` con nombre `conversation_<TIMESTAMP>.(json|csv)`.
- Opción de anonimizar: la casilla "Anonymize" redac­tará emails, números de teléfono y secuencias numéricas largas antes de guardar.

Pasos rápidos desde la UI:

1. Interactúa con el chatbot hasta que tengas la conversación deseada.
2. En el panel derecho selecciona formato `JSON` o `CSV`.
3. Marca (o desmarca) `Anonymize` según prefieras.
4. Pulsa `Create export` y luego `Download export` para guardar el archivo localmente.

### Acceso programático

Si prefieres exportar desde código (por ejemplo, al finalizar una sesión), existe un helper:

- Módulo: `chatbot/exports/conversation_exporter.py`
- Funciones: `export_conversation_json(messages, patient_state=None, metadata=None, anonymize=False, out_dir=None)` y `export_conversation_csv(messages, out_dir=None, anonymize=False)`

Ejemplo mínimo:

```py
from exports import export_conversation_json

messages = [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Hola, ¿en qué puedo ayudar?"}]
export_conversation_json(messages, anonymize=True)
```

## 🧪 Notas y buenas prácticas

- Guarda las claves y secretos en `.env` (no subir a Git).
- Los modelos grandes deberían almacenarse fuera del repositorio y cargarse mediante rutas configurables (`chatbot/config.py`).
- Si quieres que implemente una descarga en memoria (sin escribir archivos) o un botón "Export & Email", dímelo y lo implemento.
