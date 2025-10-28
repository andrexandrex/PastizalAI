# Cargar las librerías necesarias
library(shiny)
library(leaflet)
library(terra) 
library(fontawesome) # Para los íconos
library(bslib)         # Para el dashboard y las tarjetas (cards)
library(sf)            # Para leer el Shapefile

# ===================================================================
# --- Cargar el Shapefile de TamboKarkas (ANTES DE LA UI) ---
# ===================================================================
# Asegúrate de tener una carpeta 'data' con tu Shapefile
tryCatch({
  # Usamos la ruta relativa a la carpeta 'data'
  tambokarkas_sf <- sf::st_read("data/Tambokarkas5.shp") 
  
  # Reproyectar a WGS 84 (EPSG:4326) que es lo que usa Leaflet
  tambokarkas_sf_wgs84 <- sf::st_transform(tambokarkas_sf, 4326)
  
  # Opcional: Mostrar notificación si se carga bien
  # showNotification("Shapefile de TamboKarkas cargado.", type = "message")
  
}, error = function(e) {
  # Si falla, la app no se detiene, pero el mapa no tendrá el polígono
  warning(paste("No se pudo cargar 'data/Tambokarkas5.shp':", e$message))
  tambokarkas_sf_wgs84 <- NULL # Ponerlo como nulo
})


# ===================================================================
# --- Interfaz de Usuario (UI) ---
# ===================================================================
ui <- fluidPage(
  
  # 1. Activar bslib y los íconos
  theme = bs_theme(version = 5), 
  fontawesome::fa_html_dependency(),
  
  # --- Barra Superior Personalizada ---
  fluidRow(
    style = "background-color: #337ab7; color: white; padding: 10px; display: flex; align-items: center;",
    column(width = 6, 
           h3("PastizalAI", style = "margin: 0; font-weight: bold; font-size: 24px;")
    ),
    column(width = 6, 
           style = "text-align: right;",
           # Asegúrate de tener la carpeta 'www/logos_prefix/'
           img(src = "logos_prefix/Logo_SERFOR.png", height = "40px", style = "margin-left: 10px;"),
           img(src = "logos_prefix/LOGO-INCUBAGRARIA.png", height = "40px", style = "margin-left: 10px;"),
           img(src = "logos_prefix/PCM-Agricultura.png", height = "40px", style = "margin-left: 10px;")
    )
  ),
  # --- Fin de la Barra Superior ---
  
  
  # --- PANEL DE PESTAÑAS PRINCIPAL ---
  tabsetPanel(
    id = "secciones",
    type = "tabs", 
    
    # --- PESTAÑA 1: DIAGNÓSTICO ---
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
                 leafletOutput("mapa_interactivo")
               ) # fin mainPanel
             ) # fin sidebarLayout
             
    ), # --- FIN PESTAÑA 1 ---
    
    
    # --- PESTAÑA 2: ANÁLISIS ---
    tabPanel("Análisis", 
             sidebarLayout(
               sidebarPanel(
                 h4("Controles de Análisis"),
                 actionButton("procesar_mapa", "Mostrar Mapa de Políticas", icon = icon("clipboard-list"), class = "btn-info"),
                 uiOutput("descripcion_politicas")
               ), # fin sidebarPanel
               
               mainPanel(
                 tags$style(type = "text/css", "#mapa_analisis {height: 85vh !important;}"),
                 leafletOutput("mapa_analisis")
               ) # fin mainPanel
             ) # fin sidebarLayout
    ), # --- FIN PESTAÑA 2 ---
    
    
    # =======================================================
    # --- PESTAÑA 3: PRIORIZACIÓN (NUEVA UBICACIÓN) ---
    # =======================================================
    tabPanel("Priorización",
             icon = icon("dollar-sign"),
             
             layout_sidebar(
               
               # --- Controles de Priorización ---
               sidebar = sidebar(
                 title = "Optimización de Presupuesto",
                 
                 numericInput("prior_monto", "Ingrese el monto del presupuesto:", 
                              value = 23289), # Valor por defecto
                 
                 actionButton("prior_run", "Calcular Optimización", 
                              icon = icon("cogs"), class = "btn-success w-100")
                 
               ), # Fin del sidebar
               
               # --- Panel principal para el Mapa TIF ---
               mainPanel(
                 
                 layout_columns(
                   col_widths = c(8, 4), # 8 para mapa, 4 para resultados
                   
                   # Columna 1: El Mapa
                   card(
                     full_screen = TRUE,
                     card_header("Mapa de Áreas Priorizadas 🗺️"),
                     leafletOutput("prior_mapa", height = "75vh") 
                   ),
                   
                   # Columna 2: Los Resultados
                   card(
                     card_header("Resultados"),
                     uiOutput("prior_summary_box")
                   )
                   
                 ) # Fin layout_columns
                 
               ) # Fin mainPanel
               
             ) # Fin layout_sidebar
    ), # --- FIN PESTAÑA 3 ---
    
    
    # --- PESTAÑA 4: SISTEMA DE ALERTA TEMPRANA (NUEVA UBICACIÓN) ---
    tabPanel("Sistema de Alerta Temprana", 
             icon = icon("triangle-exclamation"),
             
             layout_sidebar(
               
               sidebar = sidebar(
                 title = "Controles de Monitoreo",
                 
                 card(
                   full_screen = FALSE,
                   card_header("Selección de Capa"),
                   
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
               
               layout_columns(
                 col_widths = c(4, 4, 4), 
                 
                 value_box(
                   title = "Métrica Principal",
                   value = textOutput("alerta_kpi_1_valor"),
                   showcase = icon("tachometer-alt"),
                   p(textOutput("alerta_kpi_1_titulo")) 
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
               ), # Fin layout_columns (KPIs)
               
               layout_columns(
                 col_widths = c(7, 5), 
                 
                 card(
                   full_screen = TRUE,
                   card_header("Mapa de Monitoreo 🗺️"),
                   leafletOutput("alerta_mapa", height = "600px") 
                 ),
                 card(
                   full_screen = TRUE,
                   card_header("Distribución de Valores 📊"),
                   plotOutput("alerta_grafico_histograma", height = "600px") 
                 )
               ) # Fin layout_columns (Gráfico + Mapa)
               
             ) # Fin layout_sidebar
    ), # --- FIN PESTAÑA 4 ---
    
    
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
        r_leaflet <- project(r, "EPSG:4326", method = "bilinear") 
      } else {
        r_leaflet <- r
      }
      
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
      
      pal <- colorNumeric(
        palette = c("red", "yellow", "green"),
        domain = pal_domain,
        na.color = "transparent"
      )
      
      e <- ext(r_leaflet)
      
      leafletProxy("mapa_interactivo") %>% 
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
      setView(lng = -69.763828, lat = -14.530567, zoom = 13) %>% 
      addProviderTiles(providers$CartoDB.Positron, group = "Base") %>%
      addProviderTiles(providers$OpenStreetMap.Mapnik, group = "Google Maps") %>%
      addProviderTiles(providers$Esri.WorldImagery, group = "Satélite") %>%
      addLayersControl(
        baseGroups = c("Base", "Google Maps", "Satélite"),
        overlayGroups = c("Capa TIF"), 
        options = layersControlOptions(collapsed = FALSE)
      )
  })
  
  
  # 4. Evento Reactivo: Cargar TIF de Políticas y Mostrar Texto
  observeEvent(input$procesar_mapa, {
    
    # 4.1. Mostrar el texto de las políticas
    output$descripcion_politicas <- renderUI({
      tagList(
        hr(),
        h5("Descripción de Condiciones Ecológicas:", style = "font-weight:bold;"),
        
        tags$p(tags$strong("Caso 1 (Verde):", style = "color:#32CD32;"), " Ecosistema saludable y estable."),
        tags$p(tags$strong("Caso 2 (Amarillo):", style = "color:#FFFF00;"), " Suelo sano pero vegetación reducida."),
        tags$p(tags$strong("Caso 3 (Naranja):", style = "color:#FFA500;"), " Alta cobertura, baja productividad (estrés hídrico)."), 
        tags$p(tags$strong("Caso 4 (Rojo):", style = "color:#FF8C00;"), " Pérdida de carbono inicial."),
        tags$p(tags$strong("Caso 5 (Rojo):", style = "color:#FF4500;"), " Suelo degradado con baja cobertura."),
        tags$p(tags$strong("Caso 6 (Rojo):", style = "color:#DC143C;"), " Vegetación intermitente / pisoteo alto."),
        tags$p(tags$strong("Caso 7 (Rojo):", style = "color:#8B008B;"), " Degradación severa / erosión alta.")
      ) 
    }) 
    
    # 4.2. Cargar el Mapa TIF de Políticas
    showNotification("Cargando Mapa de Políticas Propuestas...", type = "message", duration = 3)
    
    tryCatch({
      
      r <- rast("SERFOR_2024_v2_clip.tif") 
      
      if (crs(r) != "EPSG:4326") {
        showNotification("Reproyectando TIF a EPSG:4326...", type = "message")
        r_leaflet <- project(r, "EPSG:4326", method = "near")
      } else {
        r_leaflet <- r
      }
      
      colores_politicas <- c(
        "#32CD32", "#FFFF00", "#FFA500", "#FF8C00", "#FF4500", "#DC143C", "#8B008B" 
      )
      
      etiquetas_politicas <- c(
        "1 - Ecosistema saludable", "2 - Suelo sano, veg. reducida", "3 - Estrés hídrico",
        "4 - Pérdida de carbono inicial", "5 - Suelo degradado", "6 - Vegetación intermitente",
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
  # --- LÓGICA DE PESTAÑA 3: PRIORIZACIÓN ---
  # =======================================================
  
  # 5. Renderizar el mapa base de PRIORIZACIÓN (con el SHP)
  output$prior_mapa <- renderLeaflet({
    
    mapa_base <- leaflet() %>%
      setView(lng = -69.763828, lat = -14.530567, zoom = 13) %>% # Vista Picotani
      addProviderTiles(providers$CartoDB.Positron, group = "Base") %>%
      addProviderTiles(providers$Esri.WorldImagery, group = "Satélite")
    
    # Añadir el Shapefile de TamboKarkas (si se cargó)
    if (!is.null(tambokarkas_sf_wgs84)) {
      mapa_base <- mapa_base %>%
        addPolygons(
          data = tambokarkas_sf_wgs84,
          color = "#0000FF",      # Borde azul
          weight = 2,            
          fillOpacity = 0.05,    
          opacity = 1,           
          group = "Región TamboKarkas", 
          label = "Región TamboKarkas" 
        )
    }
    
    mapa_base %>%
      addLayersControl(
        baseGroups = c("Base", "Satélite"),
        overlayGroups = c("Región TamboKarkas", "Áreas Priorizadas"), 
        options = layersControlOptions(collapsed = FALSE)
      )
  })
  
  # 6. Reactivo para los resultados de Priorización
  resultados_prior <- eventReactive(input$prior_run, {
    
    monto <- input$prior_monto
    
    if (monto == 23289) {
      # --- Caso 1 ---
      tif_path <- "budget_01_selection.tif" # <-- REVISA ESTE NOMBRE
      summary_html <- tagList(
        tags$h5("Beneficio económico en 5 años:", style = "font-weight:bold;"),
        tags$p("$222,615", style = "color:green; font-size: 1.2em; margin-top: 0;"),
        tags$hr(style = "margin-top: 5px; margin-bottom: 5px;"),
        tags$strong("Área seleccionada:"), " 19996 pixeles (19.96 ha)", tags$br(),
        tags$strong("Carbono ganado total:"), " 1484.11 t CO2", tags$br(),
        tags$strong("Beneficio económico estimado:"), " $44,523"
      )
      
    } else if (monto == 89992) {
      # --- Caso 2 ---
      tif_path <- "budget_02_selection.tif" # <-- REVISA ESTE NOMBRE
      summary_html <- tagList(
        tags$h5("Beneficio económico en 5 años:", style = "font-weight:bold;"),
        tags$p("$891,475", style = "color:green; font-size: 1.2em; margin-top: 0;"),
        tags$hr(style = "margin-top: 5px; margin-bottom: 5px;"),
        tags$strong("Área seleccionada:"), " 6856 pixeles (68.56 ha)", tags$br(),
        tags$strong("Carbono ganado total:"), " 5943.18 tCO2", tags$br(),
        tags$strong("Beneficio económico estimado:"), " $178,295"
      )
      
    } else {
      # --- Caso Desconocido ---
      tif_path <- NULL 
      summary_html <- p("Monto no reconocido. Por favor, ingrese 23289 o 89992.", 
                        style = "color:red; font-weight:bold;")
      showNotification("Monto no válido. Use los valores de ejemplo.", type = "warning")
    }
    
    return(list(tif_path = tif_path, summary_html = summary_html))
  })
  
  # 7. Renderizar el cuadro de texto de Priorización
  output$prior_summary_box <- renderUI({
    resultados <- resultados_prior() 
    req(resultados) 
    return(resultados$summary_html) 
  })
  
  # 8. Observador para actualizar el MAPA TIF de Priorización
  observeEvent(resultados_prior(), {
    
    datos <- resultados_prior()
    tif_path <- datos$tif_path
    
    # Limpiar el mapa si el monto es inválido
    if (is.null(tif_path)) {
      leafletProxy("prior_mapa") %>% clearImages() %>% clearControls()
      return()
    }
    
    if (!file.exists(tif_path)) {
      showNotification(paste("Error: No se encuentra el TIF:", tif_path), type = "error")
      return()
    }
    
    showNotification("Cargando mapa de priorización...", type = "message")
    
    tryCatch({
      r <- rast(tif_path)
      
      if (crs(r) != "EPSG:4326") {
        r_leaflet <- project(r, "EPSG:4326", method = "near") 
      } else {
        r_leaflet <- r
      }
      
      pal <- colorFactor(c("transparent", "blue"), domain = c(0, 1), na.color = "transparent")
      
      leafletProxy("prior_mapa", data = r_leaflet) %>%
        clearImages() %>% 
        clearControls() %>% 
        addRasterImage(
          r_leaflet,
          colors = pal,
          opacity = 0.8,
          group = "Áreas Priorizadas" 
        ) %>%
        addLegend(
          position = "bottomright",
          colors = c("blue"),
          labels = c("Área Priorizada"),
          title = "Optimización",
          opacity = 1
        ) %>%
        # Volver a añadir el control de capas para que se vean AMBOS grupos
        addLayersControl(
          baseGroups = c("Base", "Satélite"),
          overlayGroups = c("Región TamboKarkas", "Áreas Priorizadas"),
          options = layersControlOptions(collapsed = FALSE)
        )
      
    }, error = function(e) {
      showNotification(paste("Error al cargar TIF:", e$message), type = "error")
    })
    
  }) # fin observeEvent
  
  
  # =======================================================
  # --- LÓGICA DE PESTAÑA 4: ALERTA TEMPRANA ---
  # =======================================================
  
  # 9. Mapa base de Alerta Temprana
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
  
  # 10. Reactivo para almacenar el TIF cargado (Alerta Temprana)
  raster_cargado <- eventReactive(input$alerta_mostrar_tif, {
    
    tif_path <- input$alerta_select_tif
    req(tif_path) 
    
    if (!file.exists(tif_path)) {
      showNotification(paste("Error: No se encuentra el archivo", tif_path), type = "error")
      return(NULL)
    }
    
    showNotification("Cargando y procesando TIF...", type = "message")
    
    tryCatch({
      r <- rast(tif_path)
      
      if (crs(r) != "EPSG:4326") {
        showNotification("Reproyectando TIF a EPSG:4326...", type = "message")
        
        # Usar "near" para el TIF categórico de degradación
        metodo_proj <- ifelse(tif_path == "Area_degradada2.tif", "near", "bilinear")
        r_leaflet <- project(r, "EPSG:4326", method = metodo_proj)
      } else {
        r_leaflet <- r
      }
      
      stats <- global(r_leaflet, c("mean", "max", "min"), na.rm = TRUE)
      
      # Mapeo de nombres de archivo a nombres legibles
      nombre_capa <- names(which(sapply(c(
        "Áreas Degradadas" = "Area_degradada2.tif",
        "NDVI Histórico" = "Tambokarkas5_alerta_NDVI_Historico.tif",
        "Anomalía NDVI" = "Tambokarkas5_alerta_Anomalia_NDVI.tif",
        "NDVI Actual" = "Tambokarkas5_alerta_NDVI_Actual.tif"
      ), `==`, tif_path)))
      
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
  
  # 11. Renderizar el Mapa TIF (Alerta Temprana)
  observeEvent(raster_cargado(), {
    datos <- raster_cargado()
    if (is.null(datos)) return() 
    
    r_leaflet <- datos$raster
    stats <- datos$stats
    pal_domain <- c(stats$min, stats$max)
    
    # --- Lógica de Paleta de Colores ---
    if (datos$path == "Area_degradada2.tif") {
      pal <- colorFactor(
        palette = c("green", "red"), # 0=Verde, 1=Rojo
        domain = c(0, 1), 
        na.color = "transparent"
      )
      titulo_leyenda <- "Degradación"
      
    } else if (datos$path == "Tambokarkas5_alerta_Anomalia_NDVI.tif") {
      pal <- colorNumeric(
        palette = c("red", "yellow", "green"),
        domain = pal_domain,
        na.color = "transparent"
      )
      titulo_leyenda <- "Anomalía NDVI"
      
    } else {
      pal <- colorNumeric(
        palette = c("#d73027", "#f46d43", "#fee08b", "#d9ef8b", "#a6d96a", "#1a9850"),
        domain = pal_domain,
        na.color = "transparent"
      )
      titulo_leyenda <- datos$nombre
    }
    
    leafletProxy("alerta_mapa", data = r_leaflet) %>% 
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
  
  # 12. Renderizar los KPIs (Alerta Temprana)
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
  
  
  # 13. Renderizar el Histograma (Alerta Temprana)
  output$alerta_grafico_histograma <- renderPlot({
    datos <- raster_cargado()
    if (is.null(datos)) return(NULL) 
    
    terra::hist(datos$raster, 
                main = paste("Distribución de", datos$nombre),
                xlab = "Valores",
                ylab = "Frecuencia",
                col = "darkblue",
                maxcell = 1000000) # Muestrea 1 millón de celdas
  })
  
  
} # --- FIN DEL SERVER ---


# ===================================================================
# --- Ejecutar la Aplicación ---
# ===================================================================
shinyApp(ui = ui, server = server)
