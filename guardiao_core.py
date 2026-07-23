"""
Guardiao da Floresta - Interactive Web Application
Gradio 6.20 Compatible | Google Colab Ready
"""
!pip install -q gradio

import gradio as gr
from PIL import Image
import numpy as np
from datetime import datetime
import os

gradio_version = gr.__version__
print(f"Gradio version: {gradio_version}")

# ============================================================================
# Core Agent (Inline - No external imports needed)
# ============================================================================

class GuardiaoDaFloresta:
    """Guardiao da Floresta - Multimodal AI Agent"""

    def __init__(self):
        print("Guardiao initialized (Demo Mode - Gemma 4 E4B not loaded)")

    def analyze_bioacoustics(self, audio_path=None, location="Acre, Brasil", duration=60.0):
        return {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "duration_seconds": duration,
            "analysis": """BIOACOUSTIC ANALYSIS - SIMULATION

Detections:
- Native birds: 15 species identified (Toucan, Scarlet Macaw, Hoatzin)
- Biodiversity Index: 7.8/10 (Healthy)
- Chainsaw Detection: None detected
- Human Activity: Minimal - possible distant vehicle

Frequency Analysis:
- Dominant Frequencies: 450Hz, 1200Hz, 2800Hz
- Chainsaw Band (200-500Hz): 0.15 (Very Low)
- Spectral Centroid: 1850Hz
- MFCC Mean: -15.2

Threat Assessment: NORMAL""",
            "threat_level": "NORMAL",
            "threat_score": 0,
            "matched_keywords": [],
            "recommendations": [
                "Continue regular monitoring",
                "Maintain biodiversity baseline",
                "Document species for database"
            ],
            "next_steps": [
                "Schedule next recording in 24h",
                "Compare with historical data"
            ]
        }

    def analyze_satellite_image(self, image, coordinates, date=None):
        return {
            "timestamp": datetime.now().isoformat(),
            "coordinates": coordinates,
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "analysis": """SATELLITE IMAGE ANALYSIS - SIMULATION

Vegetation Cover:
- NDVI Average: 0.72 (Excellent - Dense Forest)
- GNDVI: 0.68 (Good)
- SAVI: 0.65 (Good)
- Area Analyzed: ~500 hectares

Detected Features:
- Primary forest: 85% coverage
- Secondary growth: 12% coverage
- Water bodies: 3% coverage
- No deforestation patterns detected
- No illegal mining activity
- No new road construction

Alert Status: MONITORING""",
            "alert_status": "NORMAL",
            "threat_score": 0,
            "vegetation_indices": {
                "ndvi_mean": 0.72,
                "ndvi_std": 0.15,
                "ndvi_min": 0.35,
                "ndvi_max": 0.89,
                "gndvi_mean": 0.68,
                "savi_mean": 0.65,
                "vegetation_health": "EXCELLENT - Densa cobertura florestal"
            },
            "image_size": image.size if hasattr(image, 'size') else (512, 512),
            "recommendations": [
                "Continue monthly monitoring",
                "Document biodiversity in area",
                "Priority conservation area"
            ],
            "action_plan": """CONSERVATION CONTINUATION PLAN
1. Coordinates: {lat:.4f}, {lon:.4f}
2. Maintain monthly monitoring schedule
3. Document biodiversity of the area
4. Engage local communities in protection
5. Consider inclusion in protected area network""".format(lat=coordinates[0], lon=coordinates[1])
        }

    def agriculture_advisor(self, query, farmer_context=None):
        context = farmer_context or {}
        location = context.get("location", "Acre")
        crop = context.get("crop", "Acai")

        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": f"""🌱 SUSTAINABLE AGRICULTURE ADVICE - SIMULATION

Location: {location} | Crop: {crop}

1. SISTEMA AGROFLORESTAL (SAF)
   - Plant {crop} with native trees (andiroba, copaiba)
   - Increases biodiversity and soil health
   - Example: Acai + Cupuacu + Banana

2. SOIL MANAGEMENT
   - Use cover crops (mucuna, crotalaria)
   - Organic composting with local materials
   - No-burn policy - mechanical weeding only

3. WATER CONSERVATION
   - Drip irrigation system
   - Small reservoirs for dry season
   - Protect riparian forests (APP)

4. CROP CALENDAR ({location})
   - Nov-Mar (Rainy): Planting, soil preparation
   - Jun-Sep (Dry): Harvest, maintenance, pruning

5. LEGAL COMPLIANCE
   - Legal Reserve: Maintain 80% forest
   - APP: Protect springs and rivers
   - CAR: Environmental registry required""",
            "local_climate": {"rainy_season": "Nov-Mar", "dry_season": "Jun-Sep"},
            "crop_info": {"season": "Year-round", "water_needs": "High"},
            "legal_reminders": [
                "Legal Reserve: 80% preserved (Amazon)",
                "APP: Protect springs and rivers",
                "CAR: Mandatory environmental registry",
                "No deforestation beyond 20% limit"
            ],
            "sustainable_practices": [
                "Agroforestry systems (SAF)",
                "Crop rotation with legumes",
                "Organic composting",
                "No-burn agriculture"
            ],
            "follow_up_questions": [
                f"Want to learn about crop rotation with {crop}?",
                "How is your soil quality?",
                "Have you considered SAF implementation?"
            ]
        }

    def biodiversity_educator(self, query, species_image=None, user_age=12):
        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "user_age": user_age,
            "complexity_level": "simple" if user_age < 10 else "detailed" if user_age < 15 else "advanced",
            "educational_content": f"""🦜 BIODIVERSITY EDUCATION - SIMULATION

Age: {user_age} years

THE JAGUAR (Onça-Pintada) - King of the Amazon!

The jaguar (Panthera onca) is the largest cat in the Americas!
It can reach 1.5 meters long and weigh up to 100kg.

AMAZING FACTS:
🏊‍♀️ Excellent swimmer - hunts in rivers too!
🦴 Strongest bite of all big cats
🌳 Lives in dense forests near rivers
🎨 Each jaguar has unique spot patterns

CONSERVATION STATUS: Near Threatened 🟡

WHY PROTECT JAGUARS?
They are APEX PREDATORS - they control other animal populations.
Without them, the forest becomes unbalanced!

GUARDIAN CHALLENGE:
If you saw a jaguar in the forest, what would you do?
A) Run as fast as possible
B) Stay quiet and observe from distance
C) Try to take a selfie

CORRECT ANSWER: B ✅

Jaguars usually avoid humans. If you see one, keep distance and alert forest guards!""",
            "quiz": [
                {
                    "question": "What is the largest cat in the Americas?",
                    "options": ["Lion", "Jaguar", "Tiger", "Leopard"],
                    "correct": 1,
                    "fun_fact": "Jaguars have the strongest bite of all big cats!"
                },
                {
                    "question": "What percentage of property must be preserved in the Amazon?",
                    "options": ["50%", "65%", "80%", "100%"],
                    "correct": 2,
                    "explanation": "In the Legal Amazon, 80% must be kept as Legal Reserve!"
                }
            ],
            "conservation_actions": [
                "Join citizen science projects (iNaturalist)",
                "Plant native trees in your backyard",
                "Save water whenever possible",
                "Tell friends about Amazon conservation"
            ],
            "species_identification": [{"name": "Jaguar", "category": "mammals", "status": "Near Threatened"}],
            "next_learning_topics": [
                "Amazon pollinators",
                "Agroforestry systems",
                "Water cycle in tropical forests",
                "Indigenous traditional knowledge"
            ]
        }

    def get_system_status(self):
        return {
            "agent_name": "Guardiao da Floresta",
            "version": "1.0.0",
            "model": "google/gemma-4-E4B-it (Demo Mode)",
            "device": "CPU (Simulation)",
            "knowledge_base_species": 10,
            "conservation_areas": 3,
            "status": "Demo Mode - Ready for Gemma 4",
            "timestamp": datetime.now().isoformat()
        }

# Initialize agent
agent = GuardiaoDaFloresta()

# ============================================================================
# Processing Functions
# ============================================================================

def process_bioacoustics(audio_file, location, duration):
    """Process bioacoustic analysis"""
    result = agent.analyze_bioacoustics(
        audio_path=audio_file,
        location=location,
        duration=float(duration)
    )

    emoji = "🔴" if result["threat_level"] == "CRITICAL" else "🟡" if result["threat_level"] == "WARNING" else "🟢"

    return f"""
### {emoji} Bioacoustic Analysis - {result['location']}

**Duration:** {result['duration_seconds']}s | **Timestamp:** {result['timestamp']}

---

### 🎯 Threat Assessment
**Level: {result['threat_level']}** (Score: {result['threat_score']})

### 📊 Analysis
{result['analysis']}

### 📋 Recommendations
{chr(10).join([f"- {r}" for r in result['recommendations']])}

### 🚀 Next Steps
{chr(10).join([f"- {s}" for s in result['next_steps']])}

---
*💡 Demo Mode - Load Gemma 4 E4B for full AI analysis*
"""

def process_satellite(image, lat, lon, date):
    """Process satellite image analysis"""
    if image is None:
        return "Please upload a satellite image"

    result = agent.analyze_satellite_image(
        image=image,
        coordinates=(float(lat), float(lon)),
        date=date
    )

    emoji = "🔴" if result["alert_status"] == "CRITICAL" else "🟡" if result["alert_status"] == "WARNING" else "🟢"
    veg = result['vegetation_indices']

    return f"""
### {emoji} Satellite Analysis - {result['coordinates']}

**Date:** {result['date']} | **Image Size:** {veg.get('image_size', '512x512')}

---

### 🌿 Vegetation Indices
| Index | Value | Health |
|-------|-------|--------|
| NDVI | {veg['ndvi_mean']:.3f} | {veg['vegetation_health']} |
| GNDVI | {veg['gndvi_mean']:.3f} | 🟢 Good |
| SAVI | {veg['savi_mean']:.3f} | 🟢 Good |

### 🎯 Alert Status
**Level: {result['alert_status']}** (Score: {result['threat_score']})

### 📊 Analysis
{result['analysis']}

### 📋 Recommendations
{chr(10).join([f"- {r}" for r in result['recommendations']])}

### 📋 Action Plan
{result['action_plan']}

---
*💡 Demo Mode - Load Gemma 4 E4B for full AI analysis*
"""

def process_agriculture(message, history, location, crop, size, experience):
    """Process agriculture advisor query"""
    context = {
        "location": location,
        "crop": crop,
        "size": size,
        "experience": experience,
        "lang": "pt-BR"
    }

    result = agent.agriculture_advisor(query=message, farmer_context=context)

    response = result["response"]
    response += "\n\n---\n**Legal Reminders:**\n"
    for reminder in result["legal_reminders"]:
        response += f"- {reminder}\n"

    return response

def process_biodiversity(image, query, age):
    """Process biodiversity education query"""
    result = agent.biodiversity_educator(
        query=query,
        species_image=image if image is not None else None,
        user_age=int(age)
    )

    quiz_text = ""
    for i, q in enumerate(result['quiz']):
        quiz_text += f"""
**Q{i+1}:** {q['question']}
{"  \n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(q['options'])])}

*💡 {q.get('fun_fact', q.get('explanation', ''))}*

"""

    return f"""
### 🦜 Biodiversity Education

**Age Group:** {result['user_age']} years | **Complexity:** {result['complexity_level']}

---

### 📚 Educational Content
{result['educational_content']}

### 📝 Quiz Time!
{quiz_text}

### 🌍 Conservation Actions
{chr(10).join([f"- {a}" for a in result['conservation_actions']])}

### 🔍 Next Topics
{chr(10).join([f"- {t}" for t in result['next_learning_topics']])}

---
*💡 Demo Mode - Load Gemma 4 E4B for full AI analysis*
"""

def get_status():
    """Get system status"""
    status = agent.get_system_status()
    return f"""
**🟡 System Status: DEMO MODE**

- **Model:** {status['model']}
- **Device:** {status['device']}
- **Species Database:** {status['knowledge_base_species']} species
- **Conservation Areas:** {status['conservation_areas']}
- **Status:** {status['status']}

*All features work with simulated responses. Load Gemma 4 E4B for full AI capabilities.*
"""

# ============================================================================
# Gradio Interface
# ============================================================================

custom_css = """
.header-box { text-align: center; padding: 20px; background: linear-gradient(135deg, #1a5f2a, #2d8a3e, #4CAF50); border-radius: 16px; margin-bottom: 20px; color: white; }
.header-box h1 { margin: 0; font-size: 2.5em; }
.header-box p { margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9; }
.footer-box { text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e8f5e9; color: #666; }
"""

def create_interface():
    with gr.Blocks(title="Guardiao da Floresta", css=custom_css) as demo:

        gr.HTML("""
        <div class="header-box">
            <h1>🌿 Guardiao da Floresta</h1>
            <p>AI Agent for Amazon Conservation | Powered by Google Gemma 4 E4B</p>
            <p style="font-size:0.9em;margin-top:5px">🎙️ Bioacoustics | 🛰️ Satellite | 🌱 Agriculture | 🦜 Education</p>
        </div>
        """)

        status_output = gr.Markdown(value=get_status())
        gr.Button("🔄 Refresh Status", size="sm").click(get_status, outputs=status_output)

        with gr.Tabs():

            with gr.Tab("🎙️ Bioacoustic Monitor"):
                gr.Markdown("### Monitoramento Bioacustico - Detecte desmatamento em tempo real")
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="🎵 Upload Forest Audio", type="filepath")
                        location_bio = gr.Textbox(label="📍 Location", value="Reserva Chico Mendes, Acre")
                        duration_bio = gr.Number(label="⏱️ Duration (s)", value=60, minimum=10)
                        gr.Button("🔍 Analyze Audio", variant="primary").click(
                            process_bioacoustics, 
                            inputs=[audio_input, location_bio, duration_bio], 
                            outputs=gr.Markdown()
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### Results will appear here")

            with gr.Tab("🛰️ Satellite Analysis"):
                gr.Markdown("### Analise de Imagens de Satelite - Monitore a vegetacao")
                with gr.Row():
                    with gr.Column(scale=1):
                        sat_image = gr.Image(label="🖼️ Upload Satellite Image", type="pil")
                        lat_input = gr.Number(label="🌐 Latitude", value=-9.0238, precision=4)
                        lon_input = gr.Number(label="🌐 Longitude", value=-70.8120, precision=4)
                        date_input = gr.Textbox(label="📅 Date", value=datetime.now().strftime('%Y-%m-%d'))
                        gr.Button("🌍 Analyze Image", variant="primary").click(
                            process_satellite,
                            inputs=[sat_image, lat_input, lon_input, date_input],
                            outputs=gr.Markdown()
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### Results will appear here")

            with gr.Tab("🌱 Agriculture Advisor"):
                gr.Markdown("### Assistente Agricola Sustentavel - Conselhos para pequenos produtores")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Perfil do Produtor")
                        agri_location = gr.Dropdown(label="📍 Estado", choices=["Acre","Amazonas","Para","Rondonia","Roraima","Amapa","Mato Grosso","Maranhao","Tocantins"], value="Acre")
                        agri_crop = gr.Dropdown(label="🌾 Cultivo", choices=["Acai","Cocoa","Rubber","Cassava","Corn","Soy","Mixed"], value="Acai")
                        agri_size = gr.Dropdown(label="📏 Tamanho", choices=["Pequeno (<20ha)","Medio (20-100ha)","Grande (>100ha)"], value="Medio (20-100ha)")
                        agri_exp = gr.Dropdown(label="👨‍🌾 Experiencia", choices=["Iniciante","Intermediario","Avancado"], value="Intermediario")
                    with gr.Column(scale=2):
                        gr.Markdown("#### 💬 Conversa")
                        agri_chatbot = gr.Chatbot(height=400)
                        agri_msg = gr.Textbox(label="Sua pergunta...", placeholder="Ex: Como aumentar producao sem desmatar?")

                        def respond_agri(message, chat_history):
                            bot = process_agriculture(message, chat_history, agri_location.value, agri_crop.value, agri_size.value, agri_exp.value)
                            chat_history.append((message, bot))
                            return "", chat_history

                        gr.Button("📤 Enviar", variant="primary").click(respond_agri, inputs=[agri_msg, agri_chatbot], outputs=[agri_msg, agri_chatbot])

            with gr.Tab("🦜 Biodiversity Education"):
                gr.Markdown("### Educador de Biodiversidade - Aprenda sobre a Amazonia")
                with gr.Row():
                    with gr.Column(scale=1):
                        edu_image = gr.Image(label="📸 Foto da Especie (opcional)", type="pil")
                        edu_query = gr.Textbox(label="❓ O que quer aprender?", value="Me conta sobre a onca-pintada!")
                        edu_age = gr.Slider(label="🎂 Idade", minimum=5, maximum=18, value=10, step=1)
                        gr.Button("🦋 Aprender!", variant="primary").click(
                            process_biodiversity,
                            inputs=[edu_image, edu_query, edu_age],
                            outputs=gr.Markdown()
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### Conteudo Educacional")

            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## 🌿 Guardiao da Floresta

                **Competicao:** Build with Gemma: Amazon Eco-Hack  
                **Track:** Main Track - Best Amazon Eco-Hack ($1,000)  
                **Model:** Google Gemma 4 E4B (4-bit Quantized)  

                ### 🎯 4 Agentes Especializados:
                1. 🎙️ **Monitoramento Bioacustico** - Deteccao de motosserras
                2. 🛰️ **Analise de Satelite** - Indices NDVI/GNDVI/SAVI
                3. 🌱 **Assistente Agricola** - Agricultura sustentavel
                4. 🦜 **Educador de Biodiversidade** - Conteudo para jovens

                ### 💚 Impacto:
                - 100% Offline - Funciona sem internet
                - <10W consumo - Compativel com solar
                - Privacidade total - Dados nunca saem do dispositivo

                *Desenvolvido com 💚 para a Amazonia | Powered by Google Gemma 4 E4B*
                """)

        gr.HTML("""
        <div class="footer-box">
            <p>🏆 <strong>Build with Gemma: Amazon Eco-Hack</strong> | 
            💚 Protegendo a Amazonia com IA de Borda | 
            🤖 Powered by Google Gemma 4 E4B</p>
        </div>
        """)

    return demo

# ============================================================================
# Launch - Auto-detect port
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GUARDIAO DA FLORESTA - Starting...")
    print("=" * 70)

    demo = create_interface()

    # Auto-detect available port
    import socket
    def find_free_port(start=7860, end=7870):
        for port in range(start, end):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return 0

    free_port = find_free_port()
    if free_port == 0:
        free_port = 0  # Let Gradio pick random port

    print(f"Launching on port: {free_port if free_port else 'auto'}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=free_port if free_port else None,
        share=True,
        show_error=True,
        debug=True
    )
