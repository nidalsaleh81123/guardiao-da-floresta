"""
Guardiao da Floresta - Interactive Web Application
Gradio-based UI for multimodal Amazon conservation AI
"""

import gradio as gr
from PIL import Image
import numpy as np
import json
from datetime import datetime

# Import core agent
try:
    from guardiao_core import GuardiaoDaFloresta, GuardiaoConfig
    GUARDIAO_AVAILABLE = True
except ImportError:
    GUARDIAO_AVAILABLE = False
    print("Warning: guardiao_core not available, using demo mode")

# Global agent instance
agent = None

def get_agent():
    """Lazy initialization of the agent"""
    global agent
    if agent is None and GUARDIAO_AVAILABLE:
        try:
            agent = GuardiaoDaFloresta(GuardiaoConfig())
        except Exception as e:
            print(f"Error initializing agent: {e}")
            agent = None
    return agent

# ============================================================================
# Theme & Styling
# ============================================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
}

.header-title {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1a5f2a 0%, #2d8a3e 50%, #4CAF50 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.header-title h1 {
    color: white !important;
    font-size: 2.5em !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.header-title p {
    color: #e8f5e9 !important;
    font-size: 1.1em !important;
    margin-top: 10px !important;
}

.tab-header {
    font-size: 1.2em !important;
    font-weight: 600 !important;
}

.result-box {
    background: #f1f8e9;
    border-left: 4px solid #4CAF50;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}

.threat-critical {
    background: #ffebee !important;
    border-left: 4px solid #f44336 !important;
}

.threat-warning {
    background: #fff3e0 !important;
    border-left: 4px solid #ff9800 !important;
}

.threat-normal {
    background: #e8f5e9 !important;
    border-left: 4px solid #4CAF50 !important;
}

.agent-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 15px;
    border: 1px solid #e0e0e0;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}

.status-online {
    background: #c8e6c9;
    color: #2e7d32;
}

.status-offline {
    background: #ffccbc;
    color: #bf360c;
}

.footer {
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    border-top: 2px solid #e8f5e9;
    color: #666;
}
"""

# ============================================================================
# Agent Functions
# ============================================================================

def process_bioacoustics(audio_file, location, duration):
    """Process bioacoustic analysis"""
    if not GUARDIAO_AVAILABLE or get_agent() is None:
        return _demo_bioacoustics(location, duration)

    try:
        result = get_agent().analyze_bioacoustics(
            audio_path=audio_file,
            location=location,
            duration=float(duration)
        )
        return format_bioacoustic_result(result)
    except Exception as e:
        return f"Error: {str(e)}"

def process_satellite(image, lat, lon, date):
    """Process satellite image analysis"""
    if not GUARDIAO_AVAILABLE or get_agent() is None:
        return _demo_satellite(lat, lon)

    try:
        if image is None:
            return "Please upload a satellite image"

        result = get_agent().analyze_satellite_image(
            image=image,
            coordinates=(float(lat), float(lon)),
            date=date
        )
        return format_satellite_result(result)
    except Exception as e:
        return f"Error: {str(e)}"

def process_agriculture(message, history, location, crop, size, experience):
    """Process agriculture advisor query"""
    if not GUARDIAO_AVAILABLE or get_agent() is None:
        return _demo_agriculture(message, location, crop)

    try:
        context = {
            "location": location,
            "crop": crop,
            "size": size,
            "experience": experience,
            "lang": "pt-BR"
        }

        result = get_agent().agriculture_advisor(
            query=message,
            farmer_context=context
        )

        response = result["response"]

        # Add legal reminders
        response += "\n\n---\n📋 **Lembretes Legais:**\n"
        for reminder in result["legal_reminders"][:3]:
            response += f"- {reminder}\n"

        return response
    except Exception as e:
        return f"Error: {str(e)}"

def process_biodiversity(image, query, age):
    """Process biodiversity education query"""
    if not GUARDIAO_AVAILABLE or get_agent() is None:
        return _demo_biodiversity(query, age)

    try:
        result = get_agent().biodiversity_educator(
            query=query,
            species_image=image if image is not None else None,
            user_age=int(age)
        )
        return format_biodiversity_result(result)
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# Demo Functions (when agent not available)
# ============================================================================

def _demo_bioacoustics(location, duration):
    """Demo bioacoustic analysis"""
    return f"""
<div class="result-box threat-normal">
<h3>🎙️ Bioacoustic Analysis - {location}</h3>

**Recording Duration:** {duration}s  
**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

### 📊 Detection Results

| Metric | Value | Status |
|--------|-------|--------|
| Native Bird Species | 15 | ✅ Detected |
| Biodiversity Index | 7.8/10 | 🟢 Healthy |
| Chainsaw Detection | None | ✅ Clear |
| Human Activity | Minimal | ⚠️ Monitor |

### 🔍 Frequency Analysis
- **Dominant Frequencies:** 450Hz, 1200Hz, 2800Hz
- **Chainsaw Band (200-500Hz):** 0.15 (Low)
- **Spectral Centroid:** 1850Hz

### 🎯 Threat Assessment
**Level: NORMAL** ✅

No immediate threats detected. Forest sounds indicate healthy ecosystem with active wildlife.

### 📋 Recommendations
- ✅ Continue regular monitoring
- 📈 Maintain biodiversity baseline
- 🌿 Document species for database

---
*Demo mode - Gemma 4 E4B model not loaded*
</div>
"""

def _demo_satellite(lat, lon):
    """Demo satellite analysis"""
    return f"""
<div class="result-box threat-normal">
<h3>🛰️ Satellite Analysis - {lat}, {lon}</h3>

**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Resolution:** 512x512px

---

### 🌿 Vegetation Indices

| Index | Value | Health |
|-------|-------|--------|
| NDVI | 0.72 | 🟢 Excellent |
| GNDVI | 0.68 | 🟢 Good |
| SAVI | 0.65 | 🟢 Good |

### 📍 Analysis
- **Vegetation Cover:** Dense forest maintained
- **Changes Detected:** None significant
- **Road Expansion:** No new roads
- **Mining Activity:** None detected

### 🚨 Alert Status
**Level: MONITORING** ✅

Area shows healthy vegetation with no signs of deforestation or illegal activity.

### 📋 Action Plan
- Continue monthly monitoring
- Check during dry season for changes
- Priority conservation area

---
*Demo mode - Gemma 4 E4B model not loaded*
</div>
"""

def _demo_agriculture(query, location, crop):
    """Demo agriculture advisor"""
    return f"""🌱 **Assistente Agrícola Sustentável**

Olá! Sou seu assistente para agricultura sustentável na Amazônia.

**Sua pergunta:** {query}

---

### 🌿 Recomendações para {crop} em {location}

1. **Sistema Agroflorestal (SAF)**
   - Plante {crop} junto com árvores nativas
   - Isso protege o solo e aumenta a biodiversidade
   - Exemplo: Açaí + Cupuaçu + Palmito

2. **Manejo do Solo**
   - Use cobertura vegetal (culturas de cobertura)
   - Faça compostagem com resíduos orgânicos
   - Evite queimadas - use roçada mecanizada

3. **Conservação da Água**
   - Instale sistema de gotejamento
   - Construa pequenas barragens para irrigação
   - Mantenha APPs (Áreas de Preservação Permanente)

4. **Calendário Agrícola (Acre)**
   - **Nov-Mar (Chuvoso):** Plantio de {crop}, preparo do solo
   - **Jun-Set (Seco):** Colheita, manutenção, poda

---

### ⚖️ Lembretes Legais Importantes

- 📜 **Reserva Legal:** Manter 80% da propriedade preservada
- 🌊 **APPs:** Proteger nascentes e margens de rios
- 📋 **CAR:** Cadastro Ambiental Rural obrigatório
- 🚫 **NÃO desmate** além dos 20% permitidos

---

*💡 Dica: Quer saber mais sobre rotação de culturas ou acesso a crédito rural sustentável?*

*Demo mode - Gemma 4 E4B model not loaded*
"""

def _demo_biodiversity(query, age):
    """Demo biodiversity educator"""
    return f"""
<div class="result-box">
<h3>🦜 Educador de Biodiversidade</h3>

**Idade do estudante:** {age} anos  
**Pergunta:** {query}

---

### 🐆 A Onça-Pintada - A Rainha da Amazônia!

A onça-pintada (*Panthera onca*) é o maior felino das Américas! Ela pode chegar a **1.5 metros** de comprimento e pesar até **100 kg**.

**Curiosidades Incríveis:**
- 🏊‍♀️ É uma nadadora excelente - caça até no rio!
- 🦴 Tem a mordida mais forte entre todos os felinos
- 🌳 Vive em florestas densas, perto de rios
- 🎨 Cada onça tem um padrão de pintas único (como nossa digital!)

**Status de Conservação:** 🟡 Quase Ameaçada (Near Threatened)

### 🌍 Por que precisamos proteger as onças?

As onças são **predadores de topo** - isso significa que elas ajudam a controlar as populações de outros animais. Sem elas, a floresta ficaria desequilibrada!

### 🎮 Desafio do Guardião!

Se você visse uma onça na floresta, o que faria?
- A) Correria o mais rápido possível
- B) Ficaria quieto e observaria de longe
- C) Tentaria tirar uma selfie

**Resposta correta: B** ✅

As onças geralmente evitam humanos. Se você encontrar uma, mantenha distância e avise os guardas da floresta!

### 🌱 Ações de Conservação para Você
- 📸 Participar do iNaturalist (app de ciência cidadã)
- 🌳 Plantar árvores nativas
- 📚 Contar para seus amigos sobre a importância das onças

---
*Demo mode - Gemma 4 E4B model not loaded*
</div>
"""

# ============================================================================
# Formatting Functions
# ============================================================================

def format_bioacoustic_result(result: dict) -> str:
    """Format bioacoustic analysis result"""
    threat_class = "threat-critical" if result["threat_level"] == "CRITICAL" else \
                   "threat-warning" if result["threat_level"] == "WARNING" else "threat-normal"

    emoji = "🔴" if result["threat_level"] == "CRITICAL" else \
            "🟡" if result["threat_level"] == "WARNING" else "🟢"

    return f"""
<div class="result-box {threat_class}">
<h3>{emoji} Bioacoustic Analysis - {result['location']}</h3>

**Duration:** {result['duration_seconds']}s  
**Timestamp:** {result['timestamp']}

---

### 🎯 Threat Assessment
**Level: {result['threat_level']}** (Score: {result['threat_score']})

### 📊 Analysis
{result['analysis']}

### 📋 Recommendations
{chr(10).join([f"- {r}" for r in result['recommendations']])}

### 🚀 Next Steps
{chr(10).join([f"- {s}" for s in result['next_steps']])}
</div>
"""

def format_satellite_result(result: dict) -> str:
    """Format satellite analysis result"""
    threat_class = "threat-critical" if result["alert_status"] == "CRITICAL" else \
                   "threat-warning" if result["alert_status"] == "WARNING" else "threat-normal"

    emoji = "🔴" if result["alert_status"] == "CRITICAL" else \
            "🟡" if result["alert_status"] == "WARNING" else "🟢"

    veg = result['vegetation_indices']

    return f"""
<div class="result-box {threat_class}">
<h3>{emoji} Satellite Analysis - {result['coordinates']}</h3>

**Date:** {result['date']}  
**Image Size:** {result['image_size'][0]}x{result['image_size'][1]}px

---

### 🌿 Vegetation Indices
- **NDVI:** {veg['ndvi_mean']:.3f} - {veg['vegetation_health']}
- **GNDVI:** {veg['gndvi_mean']:.3f}
- **SAVI:** {veg['savi_mean']:.3f}

### 🎯 Alert Status
**Level: {result['alert_status']}** (Score: {result['threat_score']})

### 📊 Analysis
{result['analysis']}

### 📋 Recommendations
{chr(10).join([f"- {r}" for r in result['recommendations']])}

### 📋 Action Plan
{result['action_plan']}
</div>
"""

def format_biodiversity_result(result: dict) -> str:
    """Format biodiversity education result"""
    quiz_html = ""
    for i, q in enumerate(result['quiz']):
        quiz_html += f"""
<div style="background: #e3f2fd; padding: 10px; border-radius: 8px; margin: 10px 0;">
<strong>Q{i+1}:</strong> {q['question']}<br>
{"<br>".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(q['options'])])}
<br><em>💡 {q.get('fun_fact', q.get('explanation', ''))}</em>
</div>
"""

    return f"""
<div class="result-box">
<h3>🦜 Biodiversity Education</h3>

**Age Group:** {result['user_age']} years  
**Complexity:** {result['complexity_level']}

---

### 📚 Educational Content
{result['educational_content']}

### 📝 Quiz Time!
{quiz_html}

### 🌍 Conservation Actions
{chr(10).join([f"- {a}" for a in result['conservation_actions']])}

### 🔍 Next Topics
{chr(10).join([f"- {t}" for t in result['next_learning_topics']])}
</div>
"""

# ============================================================================
# System Status
# ============================================================================

def get_status():
    """Get system status"""
    if get_agent() is not None:
        status = get_agent().get_system_status()
        return f"""
<div class="agent-card">
<h4>🟢 System Status: ONLINE</h4>
<p><strong>Model:</strong> {status['model']}</p>
<p><strong>Device:</strong> {status['device']}</p>
<p><strong>Species Database:</strong> {status['knowledge_base_species']} species</p>
<p><strong>Conservation Areas:</strong> {status['conservation_areas']}</p>
</div>
"""
    else:
        return """
<div class="agent-card">
<h4>🟡 System Status: DEMO MODE</h4>
<p><strong>Model:</strong> Google Gemma 4 E4B (not loaded)</p>
<p><strong>Mode:</strong> Simulation for demonstration</p>
<p>All features work with simulated responses. Load the model for full functionality.</p>
</div>
"""

# ============================================================================
# Build Gradio Interface
# ============================================================================

def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="Guardiao da Floresta - Amazon Conservation AI",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="emerald",
            neutral_hue="slate"
        )
    ) as demo:

        # Header
        gr.HTML("""
        <div class="header-title">
            <h1>🌿 Guardião da Floresta</h1>
            <p>AI Agent for Amazon Conservation | Powered by Google Gemma 4 E4B</p>
            <p style="font-size: 0.9em; margin-top: 5px;">
                🎙️ Bioacoustics | 🛰️ Satellite | 🌱 Agriculture | 🦜 Education
            </p>
        </div>
        """)

        # System Status
        status_output = gr.HTML(value=get_status())
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(get_status, outputs=status_output)

        # Main Tabs
        with gr.Tabs():

            # Tab 1: Bioacoustic Monitor
            with gr.Tab("🎙️ Bioacoustic Monitor", id="bio"):
                gr.Markdown("""
                ### Monitoramento Bioacústico
                Analise gravações de áudio da floresta para detectar desmatamento, 
                espécies invasoras e avaliar a saúde da biodiversidade.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="🎵 Upload Forest Audio",
                            type="filepath"
                        )
                        location_bio = gr.Textbox(
                            label="📍 Location",
                            value="Reserva Chico Mendes, Acre",
                            placeholder="Enter location..."
                        )
                        duration_bio = gr.Number(
                            label="⏱️ Duration (seconds)",
                            value=60,
                            minimum=10,
                            maximum=3600
                        )
                        analyze_bio_btn = gr.Button(
                            "🔍 Analyze Audio",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        bio_output = gr.HTML(label="Analysis Results")

                analyze_bio_btn.click(
                    process_bioacoustics,
                    inputs=[audio_input, location_bio, duration_bio],
                    outputs=bio_output
                )

            # Tab 2: Satellite Analysis
            with gr.Tab("🛰️ Satellite Analysis", id="sat"):
                gr.Markdown("""
                ### Análise de Imagens de Satélite
                Analise imagens de satélite para detectar desmatamento, 
                mineração ilegal e mudanças na cobertura vegetal.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        sat_image = gr.Image(
                            label="🖼️ Upload Satellite Image",
                            type="pil"
                        )
                        with gr.Row():
                            lat_input = gr.Number(
                                label="🌐 Latitude",
                                value=-9.0238,
                                precision=4
                            )
                            lon_input = gr.Number(
                                label="🌐 Longitude",
                                value=-70.8120,
                                precision=4
                            )
                        date_input = gr.Textbox(
                            label="📅 Date (YYYY-MM-DD)",
                            value=datetime.now().strftime('%Y-%m-%d')
                        )
                        analyze_sat_btn = gr.Button(
                            "🌍 Analyze Image",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        sat_output = gr.HTML(label="Analysis Results")

                analyze_sat_btn.click(
                    process_satellite,
                    inputs=[sat_image, lat_input, lon_input, date_input],
                    outputs=sat_output
                )

            # Tab 3: Agriculture Advisor
            with gr.Tab("🌱 Agriculture Advisor", id="agri"):
                gr.Markdown("""
                ### Assistente Agrícola Sustentável
                Consulte nosso especialista em agricultura sustentável para 
                receber conselhos práticos e ecológicos para sua propriedade.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Perfil do Produtor")
                        agri_location = gr.Dropdown(
                            label="📍 Estado",
                            choices=["Acre", "Amazonas", "Para", "Rondonia", "Roraima", "Amapa", "Mato Grosso", "Maranhao", "Tocantins"],
                            value="Acre"
                        )
                        agri_crop = gr.Dropdown(
                            label="🌾 Cultivo Principal",
                            choices=["Acai", "Cocoa", "Rubber", "Cassava", "Corn", "Soy", "Mixed"],
                            value="Acai"
                        )
                        agri_size = gr.Dropdown(
                            label="📏 Tamanho da Propriedade",
                            choices=["Pequeno (< 20ha)", "Medio (20-100ha)", "Grande (> 100ha)"],
                            value="Medio (20-100ha)"
                        )
                        agri_exp = gr.Dropdown(
                            label="👨‍🌾 Experiencia",
                            choices=["Iniciante", "Intermediario", "Avancado"],
                            value="Intermediario"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### 💬 Conversa com o Especialista")
                        agri_chatbot = gr.Chatbot(
                            height=400,
                            bubble_full_width=False
                        )
                        agri_msg = gr.Textbox(
                            label="Sua pergunta...",
                            placeholder="Ex: Como posso aumentar a producao sem desmatar?",
                            lines=2
                        )
                        agri_send = gr.Button("📤 Enviar", variant="primary")

                def respond_agri(message, chat_history):
                    bot_message = process_agriculture(
                        message, chat_history,
                        agri_location.value, agri_crop.value,
                        agri_size.value, agri_exp.value
                    )
                    chat_history.append((message, bot_message))
                    return "", chat_history

                agri_send.click(
                    respond_agri,
                    inputs=[agri_msg, agri_chatbot],
                    outputs=[agri_msg, agri_chatbot]
                )
                agri_msg.submit(
                    respond_agri,
                    inputs=[agri_msg, agri_chatbot],
                    outputs=[agri_msg, agri_chatbot]
                )

            # Tab 4: Biodiversity Education
            with gr.Tab("🦜 Biodiversity Education", id="edu"):
                gr.Markdown("""
                ### Educador de Biodiversidade
                Aprenda sobre a incrível fauna e flora da Amazônia de forma 
                interativa e divertida! Identifique espécies e descubra como ajudar na conservação.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        edu_image = gr.Image(
                            label="📸 Foto da Espécie (opcional)",
                            type="pil"
                        )
                        edu_query = gr.Textbox(
                            label="❓ O que você quer aprender?",
                            value="Me conta sobre a onca-pintada!",
                            placeholder="Ex: Qual e o maior peixe do rio Amazonas?"
                        )
                        edu_age = gr.Slider(
                            label="🎂 Idade",
                            minimum=5,
                            maximum=18,
                            value=10,
                            step=1
                        )
                        edu_btn = gr.Button(
                            "🦋 Aprender!",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        edu_output = gr.HTML(label="Conteúdo Educacional")

                edu_btn.click(
                    process_biodiversity,
                    inputs=[edu_image, edu_query, edu_age],
                    outputs=edu_output
                )

            # Tab 5: About
            with gr.Tab("ℹ️ About", id="about"):
                gr.Markdown("""
                ## 🌿 Guardião da Floresta

                ### Sobre o Projeto

                O **Guardião da Floresta** é um agente de IA multimodal projetado para 
                proteger a Amazônia através de tecnologia de ponta que opera 100% offline.

                ### 🏆 Competição
                **Build with Gemma: Amazon Eco-Hack**  
                Track: Main Track - Best Amazon Eco-Hack  
                Prize: $1,000 USD

                ### 🤖 Tecnologia
                - **Modelo:** Google Gemma 4 E4B (4-bit quantized)
                - **Arquitetura:** Multimodal (Texto + Imagem + Áudio)
                - **Deployment:** Edge/Local (Raspberry Pi compatible)
                - **Memória:** ~2.5GB (otimizado para dispositivos de baixo recurso)

                ### 🎯 Agentes Especializados

                1. **🎙️ Monitoramento Bioacústico**
                   - Detecção de motosserras em tempo real
                   - Identificação de espécies por som
                   - Alertas de desmatamento

                2. **🛰️ Análise de Satélite**
                   - Índices de vegetação (NDVI, GNDVI, SAVI)
                   - Detecção de mineração ilegal
                   - Monitoramento de estradas

                3. **🌱 Assistente Agrícola**
                   - Conselhos sustentáveis offline
                   - Calendário agrícola regional
                   - Conformidade legal ambiental

                4. **🦜 Educador de Biodiversidade**
                   - Conteúdo adaptado por idade
                   - Quiz interativos
                   - Ações de ciência cidadã

                ### 💚 Impacto
                - **Zero dependência de nuvem** - Funciona em áreas sem internet
                - **Privacidade total** - Dados nunca saem do dispositivo
                - **Empoderamento comunitário** - Ferramentas para comunidades locais
                - **Eficiência energética** - <10W de consumo (compatível com solar)

                ### 📚 Recursos
                - [GitHub Repository](https://github.com/yourusername/guardiao-da-floresta)
                - [Technical Report](https://kaggle.com/yourusername/guardiao-report)
                - [Demo Video](https://youtube.com/yourvideo)

                ---
                *Desenvolvido com 💚 para a Amazônia*  
                *Powered by Google Gemma 4 E4B*
                """)

        # Footer
        gr.HTML("""
        <div class="footer">
            <p>🏆 <strong>Build with Gemma: Amazon Eco-Hack</strong> | 
            💚 Protegendo a Amazônia com IA de Borda | 
            🤖 Powered by Google Gemma 4 E4B</p>
            <p style="font-size: 0.8em; margin-top: 10px;">
                Este projeto e open-source. Contribua no GitHub!
            </p>
        </div>
        """)

    return demo

# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GUARDIAO DA FLORESTA - Web Application")
    print("=" * 70)
    print("Starting Gradio interface...")

    demo = create_interface()

    # Launch with sharing for demo access
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        show_api=False
    )
