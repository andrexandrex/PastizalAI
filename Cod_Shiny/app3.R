# Cargar las librerías necesarias
library(shiny)
library(leaflet)
library(terra) 
library(fontawesome) # <-- AÑADE ESTA LÍNEA
library(bslib)
#install.packages("fontawesome")
# ===================================================================
# --- Interfaz de Usuario (UI) ---
# ===================================================================
ui <- fluidPage(
  theme = bs_theme(version = 5),
  # --- Barra Superior Personalizada (Sin cambios) ---
  fluidRow(
    style = "background-color: #337ab7; color: white; padding: 10px; display: flex; align-items: center;",
    column(width = 6, 
           h3("PastizalAI", style = "margin: 0; font-weight: bold; font-size: 24px;")
    ),
    column(width = 6, # Ocupa la mitad derecha
           style = "text-align: right;", # Alinea el contenido a la derecha
           img(src = "logos_prefix/Logo_SERFOR.png", height = "40px", style = "margin-left: 10px;"),
           img(src = "logos_prefix/LOGO-INCUBAGRARIA.png", height = "40px", style = "margin-left: 10px;"),
           img(src = "logos_prefix/PCM-Agricultura.png", height = "40px", style = "margin-left: 10px;")
    )
  ),
  # --- Fin de la Barra Superior Personalizada ---
  
  
  # --- PANEL DE PESTAÑAS PRINCIPAL ---
  tabsetPanel(
    id = "secciones",
    type = "tabs", 
    
    # --- PESTAÑA 1: DIAGNÓSTICO (MODIFICADA) ---
    tabPanel("Diagnóstico", 
             
             sidebarLayout(
               
               sidebarPanel(
                 width = 3,
                 style = "margin-top: 20px;", 
                 h4("Filtros de selección"),
                 
                 shiny::selectInput("capa_tif", "Seleccione la capa TIF",
                                    choices = list(
                                      "Seleccione un TIF" = "",
                                      "Carbono Stock" = "Maps_Carbono_Stock.tif",
                                      "Forraje" = "mapa_DRY_WEIGHT_1.tif",
                                      "Erosión Potencial" = "A_erosion_2024_clip.tif"
                                    )),
                 
                 actionButton("procesar", "Calcular y Mostrar TIF", icon = icon("map-marked-alt"), class = "btn-primary"),
                 
                 hr(),
                 
                 # --- Sección de Políticas ELIMINADA de aquí ---
                 
                 tags$details(
                   open = NA,
                   tags$summary(style = "cursor: pointer; font-weight: bold;", "Información"),
                   tags$ul(
                     tags$li("Las capas TIF se cargan usando el paquete 'terra'."),
                     tags$li("La leyenda (índice) se genera dinámicamente.")
                   )
                 )
               ), # fin sidebarPanel
               
               mainPanel(
                 width = 9,
                 style = "margin-top: 20px;", 
                 
                 tags$style(type = "text/css", "#mapa_interactivo {height: 85vh !important;}"),
                 leafletOutput("mapa_interactivo") # Mapa para Diagnóstico
               ) # fin mainPanel
             ) # fin sidebarLayout
             
    ), # --- FIN PESTAÑA 1 ---
    
    
    # --- PESTAÑA 2: ANÁLISIS (MODIFICADA) ---
    tabPanel("Análisis", 
             sidebarLayout(
               sidebarPanel(
                 h4("Controles de Análisis"),
                 
                 # --- Sección de Políticas MOVIDA aquí ---
                 actionButton("procesar_mapa", "Mostrar Mapa de Políticas", icon = icon("clipboard-list"), class = "btn-info"),
                 
                 # Espacio dinámico para la descripción
                 uiOutput("descripcion_politicas")
                 
               ), # fin sidebarPanel
               
               mainPanel(
                 # Creamos un NUEVO mapa para esta pestaña
                 tags$style(type = "text/css", "#mapa_analisis {height: 85vh !important;}"),
                 leafletOutput("mapa_analisis") # Mapa para Análisis
               ) # fin mainPanel
             ) # fin sidebarLayout
    ), # --- FIN PESTAÑA 2 ---
    # =======================================================
    # --- PESTAÑA 3: SISTEMA DE ALERTA TEMPRANA (NUEVA) ---
    # =======================================================
    tabPanel("Sistema de Alerta Temprana", 
             icon = icon("triangle-exclamation"), # Ícono para la pestaña
             
             # Usamos el layout de sidebar de bslib
             layout_sidebar(
               
               # --- Controles del Dashboard ---
               sidebar = sidebar(
                 title = "Controles de Monitoreo",
                 
                 card(
                   full_screen = FALSE,
                   card_header("Selección de Capa"),
                   
                   # Selector para tus 4 TIFs
                   selectInput("alerta_select_tif", "Seleccionar capa de monitoreo:",
                               choices = c(
                                 "Seleccione..." = "",
                                 "Áreas Degradadas" = "Area_degradada2.tif",
                                 "NDVI Histórico (Línea Base)" = "Tambokarkas5_alerta_NDVI_Historico.tif",
                                 "Anomalía NDVI" = "Tambokarkas5_alerta_Anomalia_NDVI.tif",
                                 "NDVI Actual" = "Tambokarkas5_alerta_NDVI_Actual.tif"
                               )),
                   
                   actionButton("alerta_mostrar_tif", "Mostrar Capa", 
                                icon = icon("eye"), class = "btn-primary w-100")
                 )
               ), # Fin del sidebar
               
             # --- Contenido Principal del Dashboard ---
             
             # Fila para los KPIs (Cajas de Valor)
             layout_columns(
               col_widths = c(4, 4, 4), # Tres columnas
               
               value_box(
                 title = "Métrica Principal",
                 value = textOutput("alerta_kpi_1_valor"),
                 showcase = icon("tachometer-alt"),
                 p(textOutput("alerta_kpi_1_titulo")) # Título dinámico
               ),
               value_box(
                 title = "Valor Máximo",
                 value = textOutput("alerta_kpi_2_valor"),
                 showcase = icon("arrow-up"),
                 p(textOutput("alerta_kpi_2_titulo"))
               ),
               value_box(
                 title = "Valor Mínimo",
                 value = textOutput("alerta_kpi_3_valor"),
                 showcase = icon("arrow-down"),
                 p(textOutput("alerta_kpi_3_titulo"))
               )
             ), # Fin de layout_columns (KPIs)
             
             # Fila para Gráfico y Mapa
             layout_columns(
               col_widths = c(7, 5), # 70% para mapa, 50% para gráfico
               
               card(
                 full_screen = TRUE,
                 card_header("Mapa de Monitoreo 🗺️"),
                 leafletOutput("alerta_mapa", height = "600px") # Nuevo ID de mapa
               ),
               card(
                 full_screen = TRUE,
                 card_header("Distribución de Valores 📊"),
                 plotOutput("alerta_grafico_histograma", height = "600px") # Nuevo plot
               )
             )
             ) # Fin de layout_columns (Gráfico + Mapa)
             
             # Fin de layout_sidebar
    ) # --- FIN PESTAÑA 3 ---
    
  ) # --- FIN DEL tabsetPanel ---
  
) # --- FIN fluidPage ---


# ===================================================================
# --- Lógica del Servidor (Server) ---
# ===================================================================
server <- function(input, output, session) {
  
  # --- LÓGICA DE LA PESTAÑA "DIAGNÓSTICO" ---
  
  # 1. Renderizar el mapa base de DIAGNÓSTICO
  output$mapa_interactivo <- renderLeaflet({
    leaflet() %>%
      # Coordenadas de Picotani, Puno
      setView(lng = -69.763828, lat = -14.530567, zoom = 14) %>% 
      addProviderTiles(providers$CartoDB.Positron, group = "Base") %>%
      addProviderTiles(providers$OpenStreetMap.Mapnik, group = "Google Maps") %>%
      addProviderTiles(providers$Esri.WorldImagery, group = "Satélite") %>%
      addLayersControl(
        baseGroups = c("Base", "Google Maps", "Satélite"),
        overlayGroups = c("Capa TIF"),
        options = layersControlOptions(collapsed = FALSE)
      )
  })
  
  
  # 2. Evento Reactivo: Cargar TIF (Carbono, Forraje, Erosión)
  #    -> Actualiza "mapa_interactivo"
  observeEvent(input$procesar, {
    
    if (input$capa_tif == "") {
      showNotification("Por favor, seleccione una capa TIF.", type = "warning")
      return(NULL)
    }
    
    showNotification("Procesando y cargando capa TIF...", type = "message", duration = 3)
    
    tryCatch({
      r <- rast(input$capa_tif)
      
      if (crs(r) != "EPSG:4326") {
        showNotification("Reproyectando TIF a EPSG:4326...", type = "message")
        r_leaflet <- project(r, "EPSG:4326", method = "bilinear") # Bilinear para datos continuos
      } else {
        r_leaflet <- r
      }
      
      # Manejo robusto de min/max
      range_vals_mat <- global(r_leaflet, "range", na.rm = TRUE)
      
      if (!is.finite(range_vals_mat[1,1]) || !is.finite(range_vals_mat[1,2])) {
        showNotification("Error: No hay datos numéricos válidos en el ráster.", type = "error")
        return(NULL)
      }
      
      pal_domain <- c(range_vals_mat[1,1], range_vals_mat[1,2])
      
      if (pal_domain[1] == pal_domain[2]) {
        pal_domain[1] <- pal_domain[1] - 0.5
        pal_domain[2] <- pal_domain[2] + 0.5
      }
      
      # Paleta continua
      pal <- colorNumeric(
        palette = c("red", "yellow", "green"),
        domain = pal_domain,
        na.color = "transparent"
      )
      
      e <- ext(r_leaflet)
      
      leafletProxy("mapa_interactivo") %>% # <-- Actualiza el primer mapa
        clearImages() %>%
        clearControls() %>%
        addRasterImage(
          r_leaflet,
          colors = pal,
          opacity = 0.7,
          group = "Capa TIF"
        ) %>%
        addLegend(
          position = "bottomright",
          pal = pal,
          values = pal_domain, 
          title = "Índice (Valor)",
          opacity = 1
        ) %>%
        addLayersControl(
          baseGroups = c("Base", "Google Maps", "Satélite"),
          overlayGroups = c("Capa TIF"),
          options = layersControlOptions(collapsed = FALSE)
        ) %>%
        fitBounds(
          lng1 = e$xmin,
          lat1 = e$ymin,
          lng2 = e$xmax,
          lat2 = e$ymax
        )
      
      showNotification("¡Capa TIF cargada con éxito!", type = "message")
      
    }, error = function(e) {
      showNotification(
        paste("Error al cargar el TIF:", e$message),
        type = "error",
        duration = 10
      )
    })
    
  })
  
  # =======================================================
  # --- LÓGICA DE LA PESTAÑA "ANÁLISIS" ---
  # =======================================================
  
  # 3. Renderizar el mapa base de ANÁLISIS
  output$mapa_analisis <- renderLeaflet({
    leaflet() %>%
      # Misma vista inicial que el otro mapa
      setView(lng = -69.763828, lat = -14.530567, zoom = 13) %>% 
      addProviderTiles(providers$CartoDB.Positron, group = "Base") %>%
      addProviderTiles(providers$OpenStreetMap.Mapnik, group = "Google Maps") %>%
      addProviderTiles(providers$Esri.WorldImagery, group = "Satélite") %>%
      addLayersControl(
        baseGroups = c("Base", "Google Maps", "Satélite"),
        overlayGroups = c("Capa TIF"), # Aquí se cargará el mapa de políticas
        options = layersControlOptions(collapsed = FALSE)
      )
  })
  
  
  # 4. Evento Reactivo: Cargar TIF de Políticas y Mostrar Texto
  #    -> Actualiza "mapa_analisis" y "descripcion_politicas"
  observeEvent(input$procesar_mapa, {
    
    # --- 4.1. Mostrar el texto de las políticas (Colores actualizados) ---
    output$descripcion_politicas <- renderUI({
      tagList(
        hr(),
        h5("Descripción de Condiciones Ecológicas:", style = "font-weight:bold;"),
        
        tags$p(tags$strong("Caso 1 (Verde):", style = "color:#32CD32;"), " Ecosistema saludable y estable."),
        tags$p(tags$strong("Caso 2 (Amarillo):", style = "color:#FFFF00;"), " Suelo sano pero vegetación reducida."),
        tags$p(tags$strong("Caso 3 (Naranja):", style = "color:#FFA500;"), " Alta cobertura, baja productividad (estrés hídrico)."), # <-- CAMBIADO
        tags$p(tags$strong("Caso 4 (Rojo):", style = "color:#FF8C00;"), " Pérdida de carbono inicial."),
        tags$p(tags$strong("Caso 5 (Rojo):", style = "color:#FF4500;"), " Suelo degradado con baja cobertura."),
        tags$p(tags$strong("Caso 6 (Rojo):", style = "color:#DC143C;"), " Vegetación intermitente / pisoteo alto."),
        tags$p(tags$strong("Caso 7 (Rojo):", style = "color:#8B008B;"), " Degradación severa / erosión alta.")
      ) # Fin tagList
    }) # Fin renderUI
    
    # --- 4.2. Cargar el Mapa TIF de Políticas ---
    showNotification("Cargando Mapa de Políticas Propuestas...", type = "message", duration = 3)
    
    tryCatch({
      
      r <- rast("SERFOR_2024_v2_clip.tif") 
      
      if (crs(r) != "EPSG:4326") {
        showNotification("Reproyectando TIF a EPSG:4326...", type = "message")
        r_leaflet <- project(r, "EPSG:4326", method = "near")
      } else {
        r_leaflet <- r
      }
      
      # --- PALETA DE COLORES ACTUALIZADA (Caso 3 distinto) ---
      colores_politicas <- c(
        "#32CD32",  # 1: Verde Lima (Saludable)
        "#FFFF00",  # 2: Amarillo Eléctrico (Precaución 1)
        "#FFA500",  # 3: Naranja (Precaución 2) <-- CAMBIADO
        "#FF8C00",  # 4: Naranja Oscuro (Degradación inicial)
        "#FF4500",  # 5: Rojo-Naranja (Degradación media)
        "#DC143C",  # 6: Rojo Carmesí (Degradación alta)
        "#8B008B"   # 7: Magenta Oscuro (Degradación severa)
      )
      
      # Etiquetas para la leyenda (sin cambios)
      etiquetas_politicas <- c(
        "1 - Ecosistema saludable",
        "2 - Suelo sano, veg. reducida",
        "3 - Estrés hídrico",
        "4 - Pérdida de carbono inicial",
        "5 - Suelo degradado",
        "6 - Vegetación intermitente",
        "7 - Degradación severa"
      )
      
      pal_fact <- colorFactor(
        palette = colores_politicas,
        domain = c(1, 2, 3, 4, 5, 6, 7), 
        na.color = "transparent"
      )
      
      e <- ext(r_leaflet)
      
      leafletProxy("mapa_analisis") %>%
        clearImages() %>%
        clearControls() %>% 
        addRasterImage(
          r_leaflet,
          colors = pal_fact,
          opacity = 0.7,
          group = "Capa TIF"
        ) %>%
        addLegend(
          position = "bottomright",
          colors = colores_politicas, 
          labels = etiquetas_politicas, 
          title = "Mapa de Políticas",
          opacity = 1
        ) %>%
        addLayersControl(
          baseGroups = c("Base", "Google Maps", "Satélite"),
          overlayGroups = c("Capa TIF"),
          options = layersControlOptions(collapsed = FALSE)
        ) %>%
        fitBounds(
          lng1 = e$xmin,
          lat1 = e$ymin,
          lng2 = e$xmax,
          lat2 = e$ymax
        )
      
      showNotification("¡Mapa de Políticas cargado!", type = "message")
      
    }, error = function(e) {
      showNotification(
        paste("Error al cargar 'SERFOR_2024_v2_clip.tif':", e$message),
        type = "error",
        duration = 10
      )
    })
    
  }) # Fin de observeEvent
  
  # =======================================================
  # --- LÓGICA DE PESTAÑA 3: ALERTA TEMPRANA ---
  # =======================================================
  
  # 1. Mapa base de Alerta Temprana
  # Se renderiza una vez, cuando se abre la pestaña
  output$alerta_mapa <- renderLeaflet({
    leaflet() %>%
      setView(lng = -69.763828, lat = -14.530567, zoom = 13) %>% # Vista Picotani
      addProviderTiles(providers$CartoDB.Positron, group = "Base") %>%
      addProviderTiles(providers$Esri.WorldImagery, group = "Satélite") %>%
      addLayersControl(
        baseGroups = c("Base", "Satélite"),
        overlayGroups = c("Capa de Alerta"),
        options = layersControlOptions(collapsed = FALSE)
      )
  })
  
  # 2. Reactivo para almacenar el TIF cargado y sus estadísticas
  #    Esto se ejecuta CADA VEZ que se presiona el botón
  raster_cargado <- eventReactive(input$alerta_mostrar_tif, {
    
    tif_path <- input$alerta_select_tif
    req(tif_path) # Asegurarse de que se haya seleccionado un TIF
    
    if (!file.exists(tif_path)) {
      showNotification(paste("Error: No se encuentra el archivo", tif_path), type = "error")
      return(NULL)
    }
    
    showNotification("Cargando y procesando TIF...", type = "message")
    
    tryCatch({
      r <- rast(tif_path)
      
      # Reproyectar si es necesario (igual que en tu Pestaña 1)
      if (crs(r) != "EPSG:4326") {
        showNotification("Reproyectando TIF a EPSG:4326...", type = "message")
        
        # Usar "near" para capas categóricas, "bilinear" para continuas
        metodo_proj <- ifelse(tif_path == "areas_degradadas.tif", "near", "bilinear")
        r_leaflet <- project(r, "EPSG:4326", method = metodo_proj)
      } else {
        r_leaflet <- r
      }
      
      # Calcular estadísticas globales
      stats <- global(r_leaflet, c("mean", "max", "min"), na.rm = TRUE)
      
      # Obtener nombre legible para los KPIs
      nombre_capa <- names(which(sapply(c(
        "Áreas Degradadas" = "Area_degradada2.tif",
        "NDVI Histórico" = "Tambokarkas5_alerta_NDVI_Historico.tif",
        "Anomalía NDVI" = "Tambokarkas5_alerta_Anomalia_NDVI.tif",
        "NDVI Actual" = "Tambokarkas5_alerta_NDVI_Actual.tif"
      ), `==`, tif_path)))
      
      # Devolver una lista con todos los datos necesarios
      return(list(
        raster = r_leaflet, 
        stats = stats, 
        nombre = nombre_capa, 
        path = tif_path
      ))
      
    }, error = function(e) {
      showNotification(paste("Error al procesar TIF:", e$message), type = "error")
      return(NULL)
    })
  })
  
  # 3. Renderizar el Mapa TIF (se actualiza cuando 'raster_cargado' cambia)
  observeEvent(raster_cargado(), {
    datos <- raster_cargado()
    if (is.null(datos)) return() # Salir si hubo un error al cargar
    
    r_leaflet <- datos$raster
    stats <- datos$stats
    pal_domain <- c(stats$min, stats$max)
    
    # --- Lógica de Paleta de Colores ---
    # Esto es clave para que cada mapa se vea bien
    
    if (datos$path == "Area_degradada2.tif") {
      # PALETA CATEGÓRICA (asumiendo 0=No, 1=Sí)
      # ¡Ajusta los 'domain' y 'palette' si tus valores son diferentes!
      pal <- colorFactor(
        palette = c("green", "red"),
        domain = c(0, 1), 
        na.color = "transparent"
      )
      titulo_leyenda <- "Degradación"
      
    } else if (datos$path == "Tambokarkas5_alerta_Anomalia_NDVI.tif") {
      # PALETA DIVERGENTE (Rojo = Negativo, Verde = Positivo)
      pal <- colorNumeric(
        palette = c("red", "yellow", "green"),
        domain = pal_domain,
        na.color = "transparent"
      )
      titulo_leyenda <- "Anomalía NDVI"
      
    } else {
      # PALETA CONTINUA (para NDVI Histórico y Actual)
      pal <- colorNumeric(
        palette = c("#d73027", "#f46d43", "#fee08b", "#d9ef8b", "#a6d96a", "#1a9850"),
        domain = pal_domain,
        na.color = "transparent"
      )
      titulo_leyenda <- datos$nombre
    }
    
    # --- Actualizar el Mapa ---
    leafletProxy("alerta_mapa", data = r_leaflet) %>% # Usar leafletProxy
      clearImages() %>%
      clearControls() %>%
      addRasterImage(
        r_leaflet,
        colors = pal,
        opacity = 0.8,
        group = "Capa de Alerta"
      ) %>%
      addLegend(
        position = "bottomright",
        pal = pal,
        values = pal_domain,
        title = titulo_leyenda,
        opacity = 1
      )
  })
  
  # 4. Renderizar los KPIs (se actualizan cuando 'raster_cargado' cambia)
  output$alerta_kpi_1_valor <- renderText({
    datos <- raster_cargado()
    if (is.null(datos)) return("-")
    sprintf("%.3f", datos$stats$mean)
  })
  output$alerta_kpi_1_titulo <- renderText({
    datos <- raster_cargado()
    if (is.null(datos)) return("Valor Promedio")
    paste("Promedio:", datos$nombre)
  })
  
  output$alerta_kpi_2_valor <- renderText({
    datos <- raster_cargado()
    if (is.null(datos)) return("-")
    sprintf("%.3f", datos$stats$max)
  })
  output$alerta_kpi_2_titulo <- renderText({
    datos <- raster_cargado()
    if (is.null(datos)) return("Valor Máximo")
    paste("Máx:", datos$nombre)
  })
  
  output$alerta_kpi_3_valor <- renderText({
    datos <- raster_cargado()
    if (is.null(datos)) return("-")
    sprintf("%.3f", datos$stats$min)
  })
  output$alerta_kpi_3_titulo <- renderText({
    datos <- raster_cargado()
    if (is.null(datos)) return("Valor Mínimo")
    paste("Mín:", datos$nombre)
  })
  
  
  # 5. Renderizar el Histograma (se actualiza cuando 'raster_cargado' cambia)
  output$alerta_grafico_histograma <- renderPlot({
    datos <- raster_cargado()
    if (is.null(datos)) return(NULL) # No mostrar nada si no hay datos
    
    # terra::hist usa la base de R para graficar
    # 'maxcell' es importante para que no tarde mucho si el raster es gigante
    terra::hist(datos$raster, 
                main = paste("Distribución de", datos$nombre),
                xlab = "Valores",
                ylab = "Frecuencia",
                col = "darkblue",
                maxcell = 1000000) # Muestrea 1 millón de celdas
  })
  
}
# ===================================================================
# --- Ejecutar la Aplicación ---
# ===================================================================
shinyApp(ui = ui, server = server)
