# Let's load all of our required libraries
library(data.table)
library(dplyr)
library(caret)
library(xgboost)
library(pROC)
library(ggplot2)
library(ggfootball)
library(gganimate)
library(nflplotR)
library(shiny)
library(shinythemes)
library(magick)
library(DT)
library(plotly)
library(tidyr)

# 1. DATA LOADING:
games_data = fread("games.csv", select = c("gameId", "season", "week", "homeTeamAbbr", "visitorTeamAbbr"))
plays_data = fread("plays.csv", select = c(
  "gameId", "playId", "quarter", "down", "yardsToGo", "possessionTeam", 
  "defensiveTeam", "playDescription", "yardsGained", "pff_passCoverage", 
  "yardlineNumber", "preSnapHomeScore", "preSnapVisitorScore", "gameClock", 
  "offenseFormation", "absoluteYardlineNumber"
))

tracking_data_list = list()
for (week in 1:9) {
  file_path = paste0("tracking_week_", week, ".csv")
  week_data = fread(file_path)
  tracking_data_list[[week]] = week_data
}
tracking_data = rbindlist(tracking_data_list)

# 2. MERGE DATA: Here, we combine the plays and games datasets into one table.
merged_data = merge(plays_data, games_data, by = "gameId", all.x = TRUE)

# We'll exclude any plays that are labeled as "Predictable" (Prevent, Goal Line, or Misc).
merged_data = merged_data |>
  filter(!pff_passCoverage %in% c("Prevent", "Goal Line", "Miscellaneous"))


# 3. DEFINE OFFENSIVE & DEFENSIVE SUCCESS:
merged_data = merged_data |>
  mutate(
    offensive_success = ifelse(
      (down == 1 & yardsGained >= 5) |
        (down %in% c(2, 3, 4) & yardsGained >= yardsToGo) |
        grepl("touchdown", playDescription, ignore.case = TRUE),
      1, 0
    ),
    defensive_success = ifelse(
      (down == 1 & yardsGained < 5) |
        (down %in% c(2, 3, 4) & yardsGained < yardsToGo) |
        grepl("interception|sack|fumble", playDescription, ignore.case = TRUE),
      1, 0
    ),
    touchdown = ifelse(grepl("touchdown", playDescription, ignore.case = TRUE), 1, 0),
    turnover  = ifelse(grepl("interception|fumble", playDescription, ignore.case = TRUE), 1, 0)
  )

# 4. MERGE PLAY DATA WITH TRACKING DATA: 
tracking_with_teams = merge(
  tracking_data,
  plays_data[, .(gameId, playId, possessionTeam, defensiveTeam, offenseFormation)],
  by = c("gameId", "playId"),
  all.x = TRUE
)

tracking_with_teams = merge(
  tracking_with_teams,
  merged_data[, .(gameId, playId, yardlineNumber, preSnapHomeScore, preSnapVisitorScore, 
                  offensive_success, defensive_success, touchdown, turnover, 
                  pff_passCoverage, offenseFormation)],
  by = c("gameId", "playId"),
  all.x = TRUE
)

# 5. ENHANCED DEFENSIVE & OFFENSIVE FEATURES:
defensive_summary = tracking_with_teams |>
  group_by(gameId, playId) |>
  filter(club == defensiveTeam) |>
  summarize(
    avg_defender_speed = mean(s, na.rm = TRUE),  
    avg_defender_acceleration = mean(a, na.rm = TRUE),
    avg_defender_direction = mean(dir, na.rm = TRUE),  
    defender_spacing = max(x, na.rm = TRUE) - min(x, na.rm = TRUE),  
    total_defensive_motion = sum(event == "motion", na.rm = TRUE),  
    avg_defender_position_x = mean(x, na.rm = TRUE),
    .groups = "drop"
  )

offensive_summary = tracking_with_teams |>
  group_by(gameId, playId) |>
  filter(club == possessionTeam) |>
  summarize(
    avg_offensive_speed = mean(s, na.rm = TRUE),
    avg_offensive_acceleration = mean(a, na.rm = TRUE),
    avg_offensive_direction = mean(dir, na.rm = TRUE),  
    offensive_spacing = max(x, na.rm = TRUE) - min(x, na.rm = TRUE),  
    total_offensive_motion = sum(event == "motion", na.rm = TRUE),  
    avg_offensive_position_x = mean(x, na.rm = TRUE),
    .groups = "drop"
  )

# 6. MERGE TRACKING FEATURES INTO MERGED DATA: 
full_data = merge(merged_data, defensive_summary, by = c("gameId", "playId"), all.x = TRUE)
full_data = merge(full_data, offensive_summary, by = c("gameId", "playId"), all.x = TRUE)


# 7. ADD RELATIVE & SITUATIONAL FEATURES:
full_data = full_data |>
  mutate(
    avg_rel_acceleration = avg_offensive_acceleration - avg_defender_acceleration,
    avg_rel_direction = avg_offensive_direction - avg_defender_direction,
    spacing_diff = offensive_spacing - defender_spacing,
    total_motion_diff = total_offensive_motion - total_defensive_motion
  ) |>
  mutate(
    score_differential = ifelse(homeTeamAbbr == possessionTeam,
                                as.numeric(preSnapHomeScore - preSnapVisitorScore),
                                as.numeric(preSnapVisitorScore - preSnapHomeScore)),
    time_pressure = ifelse(quarter >= 4 & gameClock < 5*60, 1, 0)
  )

# We drop any rows with missing values to keep things clean.
full_data = na.omit(full_data)

# Converting key outcome columns into factors.
full_data$offensive_success = as.factor(full_data$offensive_success)
full_data$defensive_success = as.factor(full_data$defensive_success)
full_data$touchdown         = as.factor(full_data$touchdown)
full_data$turnover          = as.factor(full_data$turnover)


# 8. TRAIN/TEST SPLIT:
set.seed(123)
trainIndex = createDataPartition(full_data$offensive_success, p = 0.7, list = FALSE)
train_data = full_data[trainIndex, ]
test_data  = full_data[-trainIndex, ]

# Convert some columns to factors in both training and testing sets.
train_data = train_data |>
  mutate(across(c(down, possessionTeam, defensiveTeam, pff_passCoverage), as.factor))
test_data = test_data |>
  mutate(across(c(down, possessionTeam, defensiveTeam, pff_passCoverage), as.factor))

# We ensure test data has the same factor levels as training data to avoid errors.
train_data_levels = lapply(train_data |>
                             select(where(is.factor)), levels)
test_data = test_data |>
  mutate(across(
    names(train_data_levels),
    ~ factor(., levels = train_data_levels[[cur_column()]])
  ))

# 9. DEFINE FEATURE COLUMNS:
feature_cols = c("down", "yardsToGo", "avg_defender_speed", 
                 "avg_defender_acceleration", "avg_defender_direction", "defender_spacing", 
                 "total_defensive_motion", "avg_defender_position_x", "avg_offensive_speed",
                 "avg_offensive_acceleration", "avg_offensive_direction", "offensive_spacing",
                 "total_offensive_motion", "avg_offensive_position_x", "avg_rel_acceleration",
                 "avg_rel_direction", "spacing_diff", "total_motion_diff",
                 "score_differential", "time_pressure", "pff_passCoverage")


# 10. BUILD XGBOOST MODELS:
### Offensive Success
x_train_off = model.matrix(offensive_success ~ down + yardsToGo + avg_defender_speed + 
                             avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                             total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                             avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                             total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                             avg_rel_direction + spacing_diff + total_motion_diff +
                             score_differential + time_pressure + pff_passCoverage, 
                           data = train_data)[, -1]
y_train_off = as.numeric(train_data$offensive_success) - 1

x_test_off = model.matrix(offensive_success ~ down + yardsToGo + avg_defender_speed + 
                            avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                            total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                            avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                            total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                            avg_rel_direction + spacing_diff + total_motion_diff +
                            score_differential + time_pressure + pff_passCoverage, 
                          data = test_data)[, -1]
y_test_off = as.numeric(test_data$offensive_success) - 1

dtrain_off = xgb.DMatrix(data = x_train_off, label = y_train_off)
dtest_off  = xgb.DMatrix(data = x_test_off,  label = y_test_off)

cv_off = xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_off,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 10,
  verbose = 1
)
optimal_nrounds_off = cv_off$best_iteration

xgb_model_off = xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_off,
  nrounds = optimal_nrounds_off,
  verbose = 1
)
### Defensive Success
x_train_def = model.matrix(defensive_success ~ down + yardsToGo + avg_defender_speed + 
                             avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                             total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                             avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                             total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                             avg_rel_direction + spacing_diff + total_motion_diff +
                             score_differential + time_pressure + pff_passCoverage,
                           data = train_data)[, -1]
y_train_def = as.numeric(train_data$defensive_success) - 1

x_test_def = model.matrix(defensive_success ~ down + yardsToGo + avg_defender_speed + 
                            avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                            total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                            avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                            total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                            avg_rel_direction + spacing_diff + total_motion_diff +
                            score_differential + time_pressure + pff_passCoverage, 
                          data = test_data)[, -1]
y_test_def = as.numeric(test_data$defensive_success) - 1

dtrain_def = xgb.DMatrix(data = x_train_def, label = y_train_def)
dtest_def  = xgb.DMatrix(data = x_test_def,  label = y_test_def)

cv_def = xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_def,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 10,
  verbose = 1
)
optimal_nrounds_def = cv_def$best_iteration

xgb_model_def = xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_def,
  nrounds = optimal_nrounds_def,
  verbose = 1
)
### Yards Gained (Regression)
x_train_yards = model.matrix(yardsGained ~ down + yardsToGo + avg_defender_speed + 
                               avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                               total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                               avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                               total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                               avg_rel_direction + spacing_diff + total_motion_diff +
                               score_differential + time_pressure + pff_passCoverage, 
                             data = train_data)[, -1]
y_train_yards = train_data$yardsGained

x_test_yards = model.matrix(yardsGained ~ down + yardsToGo + avg_defender_speed + 
                              avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                              total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                              avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                              total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                              avg_rel_direction + spacing_diff + total_motion_diff +
                              score_differential + time_pressure + pff_passCoverage,
                            data = test_data)[, -1]
y_test_yards = test_data$yardsGained

dtrain_yards = xgb.DMatrix(data = x_train_yards, label = y_train_yards)
dtest_yards  = xgb.DMatrix(data = x_test_yards,  label = y_test_yards)

cv_yards = xgb.cv(
  params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_yards,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 10,
  verbose = 1
)
optimal_nrounds_yards = cv_yards$best_iteration

xgb_model_yards = xgb.train(
  params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_yards,
  nrounds = optimal_nrounds_yards,
  verbose = 1
)
### Touchdown Probability (Binary)
x_train_td = model.matrix(touchdown ~ down + yardsToGo + avg_defender_speed + 
                            avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                            total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                            avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                            total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                            avg_rel_direction + spacing_diff + total_motion_diff +
                            score_differential + time_pressure + pff_passCoverage, 
                          data = train_data)[, -1]
y_train_td = as.numeric(train_data$touchdown) - 1

x_test_td = model.matrix(touchdown ~ down + yardsToGo + avg_defender_speed + 
                           avg_defender_acceleration + avg_defender_direction + defender_spacing + 
                           total_defensive_motion + avg_defender_position_x + avg_offensive_speed +
                           avg_offensive_acceleration + avg_offensive_direction + offensive_spacing +
                           total_offensive_motion + avg_offensive_position_x + avg_rel_acceleration +
                           avg_rel_direction + spacing_diff + total_motion_diff +
                           score_differential + time_pressure + pff_passCoverage, 
                         data = test_data)[, -1]
y_test_td = as.numeric(test_data$touchdown) - 1

dtrain_td = xgb.DMatrix(data = x_train_td, label = y_train_td)
dtest_td  = xgb.DMatrix(data = x_test_td,  label = y_test_td)

cv_td = xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_td,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 10,
  verbose = 1
)
optimal_nrounds_td = cv_td$best_iteration

xgb_model_td = xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_td,
  nrounds = optimal_nrounds_td,
  verbose = 1
)

# 11. EVALUATE MODELS:
evaluate_classification = function(model, dtest, y_test, title_suffix="") {
  preds = predict(model, dtest)
  roc_obj = roc(y_test, preds)
  auc_val = auc(roc_obj)
  plot(roc_obj, main = paste("ROC Curve - AUC:", round(auc_val,2), title_suffix), col = "blue")
  return(auc_val)
}

evaluate_regression = function(model, dtest, y_test, title_suffix="") {
  preds = predict(model, dtest)
  rmse_val = sqrt(mean((preds - y_test)^2))
  plot(y_test, preds,
       main = paste("Actual vs Predicted Yards Gained - RMSE:", round(rmse_val,2), title_suffix),
       xlab = "Actual Yards Gained", ylab = "Predicted Yards Gained", pch = 19, col = "blue")
  abline(0,1,col="red",lwd=2)
  return(rmse_val)
}

auc_off  = evaluate_classification(xgb_model_off, dtest_off, y_test_off, "Offensive Success")
auc_def  = evaluate_classification(xgb_model_def, dtest_def, y_test_def, "Defensive Success")
auc_td   = evaluate_classification(xgb_model_td,  dtest_td,  y_test_td,  "Touchdown Probability")
rmse_yards = evaluate_regression(xgb_model_yards, dtest_yards, y_test_yards, "Yards Gained")
# 12. TEAM COLORS & SHINY APP:
# This is a simple mapping of teams to their colors for visual consistency
team_colors = list(
  "BAL" = "#241773", "BUF" = "#00338D", "CIN" = "#FB4F14", "CLE" = "#311D00",
  "DEN" = "#FB4F14", "HOU" = "#03202F", "IND" = "#002C5F", "JAX" = "#006778",
  "KC"  = "#E31837", "LV"  = "#000000", "LAC" = "#0080C6", "MIA" = "#008E97",
  "NE"  = "#002244", "NYJ" = "#125740", "PIT" = "#FFB612", "TEN" = "#4B92DB",
  "ARI" = "#97233F", "ATL" = "#A71930", "CAR" = "#0085CA", "CHI" = "#C83803",
  "DAL" = "#808080", "DET" = "#0076B6", "GB"  = "#203731", "LAR" = "#003594",
  "MIN" = "#4F2683", "NO"  = "#D3BC8D", "NYG" = "#0B2265", "PHI" = "#004C54",
  "SF"  = "#AA0000", "SEA" = "#002244", "TB"  = "#D50A0A", "WAS" = "#773141"
)

ui = fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("NFL Official Scouting Report Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("team", "Select Team:", choices = unique(full_data$possessionTeam)),
      selectInput("gameId", "Select Game ID:", choices = NULL),
      selectInput("playId", "Select Play ID:", choices = NULL),
      sliderInput("fps", "Animation Speed (Frames per Second):", min = 1, max = 20, value = 5),
      downloadButton("downloadMetrics", "Download Player Metrics"),
      actionButton("run", "Generate Insights", icon=icon("play"))
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Animation", imageOutput("playAnimation")),
        tabPanel("Play Description", tableOutput("playDescription")),
        
        tabPanel("Historical Data", 
                 h3("Offensive Success Against Coverages"), 
                 plotOutput("offensiveSuccess"),
                 h3("Defensive Usage and Success"), 
                 plotOutput("defensiveSuccess"),
                 h3("Team-Wide Predictions"),
                 tableOutput("teamPredictionSummary")),
        
        tabPanel("Situational Analysis",
                 h3("Visualizing Offensive Success"),
                 plotOutput("offensiveDownDistanceChart"),
                 h3("Visualizing Defensive Success"),
                 plotOutput("defensiveDownDistanceChart")),
        
        tabPanel("Yards Gained Prediction",
                 h3("Actual vs Predicted Yards Gained"),
                 plotOutput("yardsGainedPlot")),
        
        tabPanel("Touchdown Probability",
                 h3("Touchdown Probability for Selected Play"),
                 plotlyOutput("touchdownProbabilityPlot")),
      )
    )
  )
)

server = function(input, output, session) {
  
  # Dynamically update gameId and playId selections based on the chosen team.
  observe({
    updateSelectInput(session, "gameId",
                      choices = unique(full_data |>
                                         filter(possessionTeam == input$team | defensiveTeam == input$team) |>
                                         pull(gameId)))
  })
  observe({
    updateSelectInput(session, "playId",
                      choices = unique(full_data |>
                                         filter(gameId == input$gameId) |>
                                         pull(playId)))
  })
  
  # Animation Tab:
  animation_trigger = reactiveVal(FALSE)
  observeEvent(input$run, { animation_trigger(TRUE) })
  
  output$playAnimation = renderImage({
    req(animation_trigger(), input$gameId, input$playId)
    
    play_data = tracking_data |>
      filter(gameId == input$gameId, playId == input$playId)
    
    play_info = full_data |>
      filter(gameId == input$gameId, playId == input$playId)
    
    if (!"club" %in% names(play_data)) {
      showNotification("No 'club' column; color fill may not work.", type="error")
    }
    if (!"frameId" %in% names(play_data)) {
      showNotification("No 'frameId' column; animation won't progress over time.", type="error")
    }
    
    home_score  = unique(play_info$preSnapHomeScore)
    visitor_score = unique(play_info$preSnapVisitorScore)
    time_left   = unique(play_info$gameClock)
    coverage    = unique(play_info$pff_passCoverage)
    play_result = unique(play_info$playDescription)
    
    p_anim = ggfootball(left_endzone="red", right_endzone="blue", field_alpha=0.7) +
      geom_point(
        data = play_data,
        aes(x = x, y = y, fill = club),
        size=8, shape=21, color="black"
      ) +
      geom_text(
        data = play_data,
        aes(x = x, y = y, label=jerseyNumber),
        size=3, color="white", vjust=0.5
      ) +
      scale_fill_nfl(type="primary") +
      labs(
        title = paste("Home", home_score, "-", visitor_score, 
                      "| Time:", time_left),
        subtitle = paste("Coverage:", coverage),
        caption = paste("Play Result:", play_result),
        x = "Yard Line",
        y = "Width (yards)"
      ) +
      theme_minimal() +
      theme(
        legend.position="none",
        plot.title=element_text(size=14, face="bold", hjust=0.5),
        plot.subtitle=element_text(size=12, hjust=0.5),
        plot.caption=element_text(size=10, hjust=0.5)
      ) +
      transition_time(frameId) +
      ease_aes("linear")
    
    gif_file = tempfile(fileext=".gif")
    animate(p_anim, nframes=100, fps=input$fps, renderer=gifski_renderer(gif_file))
    
    list(src = gif_file, contentType="image/gif", alt="Play Animation")
  }, deleteFile=TRUE)
  
  # Play Description Tab: Show a small table describing the play.
  output$playDescription = renderTable({
    req(input$gameId, input$playId)
    full_data |>
      filter(gameId == input$gameId, playId == input$playId) |>
      select(gameId, playId, possessionTeam, defensiveTeam, playDescription)
  })
  # Historical Data Tab - Offensive Success Plot
  output$offensiveSuccess = renderPlot({
    req(input$team)
    data = full_data |>
      filter(possessionTeam == input$team) |>
      group_by(pff_passCoverage) |>
      summarize(success_rate = mean(as.numeric(as.character(offensive_success)), na.rm = TRUE) * 100,
                n = n())
    
    ggplot(data, aes(x = pff_passCoverage, y = success_rate, fill = input$team)) +
      geom_bar(stat = "identity", fill = team_colors[[input$team]]) +
      geom_text(aes(label = n), vjust = -0.5) +
      labs(title = paste("Offensive Success Against Coverages:", input$team),
           x = "Coverage Type", y = "Success Rate (%)") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  # Historical Data Tab - Defensive Success Plot
  output$defensiveSuccess = renderPlot({
    req(input$team)
    data = full_data |>
      filter(defensiveTeam == input$team) |>
      group_by(pff_passCoverage) |>
      summarize(success_rate = mean(as.numeric(as.character(defensive_success)), na.rm = TRUE) * 100,
                n = n())
    
    ggplot(data, aes(x = pff_passCoverage, y = success_rate, fill = input$team)) +
      geom_bar(stat = "identity", fill = team_colors[[input$team]]) +
      geom_text(aes(label = n), vjust = -0.5) +
      labs(title = paste("Defensive Usage and Success:", input$team),
           x = "Coverage Type", y = "Success Rate (%)") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  # Historical Data Tab - Team-Wide Predictions
  output$teamPredictionSummary = renderTable({
    req(input$gameId)
    
    # We gather the two teams playing from the chosen game.
    game_teams = full_data |>
      filter(gameId == input$gameId) |>
      select(possessionTeam, defensiveTeam) |>
      distinct() |>
      unlist() |>
      unique()
    
    team_data = full_data |>
      filter(possessionTeam %in% game_teams | defensiveTeam %in% game_teams)
    
    if (nrow(team_data) == 0) {
      return(data.frame(message = "No data for the selected game"))
    }
    # Keep factor alignment consistent with training data if it exists in the environment.
    if (exists("train_data")) {
      team_data = team_data |>
        mutate(across(
          c(down, pff_passCoverage),
          ~ factor(., levels = levels(train_data[[cur_column()]]))
        ))
    }
    # Define feature matrix for Offensive Success
    feature_matrix_off = tryCatch({
      model.matrix(
        ~ down + yardsToGo + avg_defender_speed + avg_defender_acceleration +
          avg_defender_direction + defender_spacing + total_defensive_motion +
          avg_defender_position_x + avg_offensive_speed + avg_offensive_acceleration +
          avg_offensive_direction + offensive_spacing + total_offensive_motion +
          avg_offensive_position_x + avg_rel_acceleration + avg_rel_direction +
          spacing_diff + total_motion_diff + score_differential + time_pressure +
          pff_passCoverage, 
        data = team_data
      )[, -1]
    }, error = function(e) {
      stop("Error in creating feature matrix for Offensive Success: ", e$message)
    })
    # Predictions for Offensive Success
    dmatrix_off = xgb.DMatrix(data = feature_matrix_off)
    predictions_off = predict(xgb_model_off, dmatrix_off)
    team_data$predicted_off_success = round(predictions_off, 2)
    
    # Define feature matrix for Defensive Success
    feature_matrix_def = tryCatch({
      model.matrix(
        ~ down + yardsToGo + avg_defender_speed + avg_defender_acceleration +
          avg_defender_direction + defender_spacing + total_defensive_motion +
          avg_defender_position_x + avg_offensive_speed + avg_offensive_acceleration +
          avg_offensive_direction + offensive_spacing + total_offensive_motion +
          avg_offensive_position_x + avg_rel_acceleration + avg_rel_direction +
          spacing_diff + total_motion_diff + score_differential + time_pressure +
          pff_passCoverage, 
        data = team_data
      )[, -1]
    }, error = function(e) {
      stop("Error in creating feature matrix for Defensive Success: ", e$message)
    })
    
    # Predictions for Defensive Success
    dmatrix_def = xgb.DMatrix(data = feature_matrix_def)
    predictions_def = predict(xgb_model_def, dmatrix_def)
    team_data$predicted_def_success = round(predictions_def, 2)
    
    # Define feature matrix for Yards Gained
    feature_matrix_yards = tryCatch({
      model.matrix(
        ~ down + yardsToGo + avg_defender_speed + avg_defender_acceleration +
          avg_defender_direction + defender_spacing + total_defensive_motion +
          avg_defender_position_x + avg_offensive_speed + avg_offensive_acceleration +
          avg_offensive_direction + offensive_spacing + total_offensive_motion +
          avg_offensive_position_x + avg_rel_acceleration + avg_rel_direction +
          spacing_diff + total_motion_diff + score_differential + time_pressure +
          pff_passCoverage, 
        data = team_data
      )[, -1]
    }, error = function(e) {
      stop("Error in creating feature matrix for Yards Gained: ", e$message)
    })
    
    # Predictions for Yards Gained
    dmatrix_yards = xgb.DMatrix(data = feature_matrix_yards)
    predictions_yards = predict(xgb_model_yards, dmatrix_yards)
    team_data$predicted_yards_gained = round(predictions_yards, 2)
    
    # Define feature matrix for Touchdown Probability
    feature_matrix_td = tryCatch({
      model.matrix(
        ~ down + yardsToGo + avg_defender_speed + avg_defender_acceleration +
          avg_defender_direction + defender_spacing + total_defensive_motion +
          avg_defender_position_x + avg_offensive_speed + avg_offensive_acceleration +
          avg_offensive_direction + offensive_spacing + total_offensive_motion +
          avg_offensive_position_x + avg_rel_acceleration + avg_rel_direction +
          spacing_diff + total_motion_diff + score_differential + time_pressure +
          pff_passCoverage, 
        data = team_data
      )[, -1]
    }, error = function(e) {
      stop("Error in creating feature matrix for Touchdown Probability: ", e$message)
    })
    
    # Predictions for Touchdown Probability
    dmatrix_td = xgb.DMatrix(data = feature_matrix_td)
    predictions_td = predict(xgb_model_td, dmatrix_td)
    team_data$predicted_td_prob = round(predictions_td, 4)
    
    # We'll group the predictions by team. We typically only have two teams per game.
    team_predictions = team_data |>
      group_by(possessionTeam) |>
      summarize(
        avg_predicted_off_success = mean(predicted_off_success, na.rm = TRUE),
        avg_predicted_def_success = mean(predicted_def_success, na.rm = TRUE),
        avg_predicted_yards = mean(predicted_yards_gained, na.rm = TRUE),
        avg_predicted_td_prob = mean(predicted_td_prob, na.rm = TRUE),
        total_plays = n(),
        .groups = 'drop'
      ) |>
      arrange(desc(avg_predicted_off_success))
    
    # Make sure we only show the two teams that played.
    if (length(game_teams) > 2) {
      game_teams = game_teams[1:2]
    }
    
    team_predictions = team_predictions |>
      filter(possessionTeam %in% game_teams)
    
    team_predictions
  })
  
  # Situational Analysis Tab - Offensive Down Distance Chart
  output$offensiveDownDistanceChart = renderPlot({
    req(input$team)
    
    # Break down the field into 20-yard chunks, group by down, and compute success rate.
    data = full_data |>
      filter(possessionTeam == input$team) |>
      mutate(
        offensive_success = as.numeric(as.character(offensive_success)),
        yard_chunk = cut(
          absoluteYardlineNumber, 
          breaks = c(0, 20, 40, 60, 80, 100), 
          include.lowest = TRUE,
          labels = c("10-29", "30-49", "50-69", "70-89", "90-110")
        )
      ) |>
      group_by(yard_chunk, down) |>
      summarize(
        success_rate = mean(offensive_success, na.rm = TRUE) * 100,
        total_plays = n(),
        .groups = 'drop'
      ) |>
      complete(yard_chunk, down, fill = list(success_rate = 0, total_plays = 0))
    
    data = data |>
      mutate(
        x_coord = case_when(
          yard_chunk == "10-29" ~ 20,
          yard_chunk == "30-49" ~ 40,
          yard_chunk == "50-69" ~ 60,
          yard_chunk == "70-89" ~ 80,
          yard_chunk == "90-110" ~ 100
        ),
        y_coord = case_when(
          down == 1 ~ 15,
          down == 2 ~ 30,
          down == 3 ~ 45
        ),
        down_label = case_when(
          down == 1 ~ "1st Down",
          down == 2 ~ "2nd Down",
          down == 3 ~ "3rd Down"
        )
      )
    
    ggfootball(left_endzone = "red", right_endzone = "blue", field_alpha = 0.7) +
      geom_text(
        data = data,
        aes(
          x = x_coord, 
          y = y_coord, 
          label = ifelse(total_plays > 0, 
                         paste0(down_label, "\n", round(success_rate, 1), "%\n(", total_plays, " plays)"), 
                         "No Data")
        ),
        size = 4, color = "black", fontface = "bold"
      ) +
      geom_text(
        aes(x = 5, y = 25, label = "OWN\nEND ZONE"),
        angle = 90, size = 6, color = "white", fontface = "bold"
      ) +
      geom_text(
        aes(x = 115, y = 25, label = "OPPONENT\nEND ZONE"),
        angle = 90, size = 6, color = "white", fontface = "bold"
      ) +
      labs(
        title = paste("Measuring Offensive Success by Down and Field Position (20 yard chunks)", input$team),
        x = "Yard Line",
        y = "Field Width (yards)"
      ) +
      theme_minimal()
  })
  
  # Situational Analysis Tab - Defensive Down Distance Chart
  output$defensiveDownDistanceChart = renderPlot({
    req(input$team)
    
    # Similar approach for defensive success rate.
    data = full_data |>
      filter(defensiveTeam == input$team) |>
      mutate(
        defensive_success = as.numeric(as.character(defensive_success)),
        yard_chunk = cut(
          absoluteYardlineNumber, 
          breaks = c(0, 20, 40, 60, 80, 100), 
          include.lowest = TRUE,
          labels = c("10-29", "30-49", "50-69", "70-89", "90-110")
        )
      ) |>
      group_by(yard_chunk, down) |>
      summarize(
        success_rate = mean(defensive_success, na.rm = TRUE) * 100,
        total_plays = n(),
        .groups = 'drop'
      ) |>
      complete(yard_chunk, down, fill = list(success_rate = 0, total_plays = 0))
    
    data = data |>
      mutate(
        x_coord = case_when(
          yard_chunk == "10-29" ~ 20,
          yard_chunk == "30-49" ~ 40,
          yard_chunk == "50-69" ~ 60,
          yard_chunk == "70-89" ~ 80,
          yard_chunk == "90-110" ~ 100
        ),
        y_coord = case_when(
          down == 1 ~ 15,
          down == 2 ~ 30,
          down == 3 ~ 45
        ),
        down_label = case_when(
          down == 1 ~ "1st Down",
          down == 2 ~ "2nd Down",
          down == 3 ~ "3rd Down"
        )
      )
    
    ggfootball(left_endzone = "red", right_endzone = "blue", field_alpha = 0.7) +
      geom_text(
        data = data,
        aes(
          x = x_coord, 
          y = y_coord, 
          label = ifelse(total_plays > 0, 
                         paste0(down_label, "\n", round(success_rate, 1), "%\n(", total_plays, " plays)"), 
                         "No Data")
        ),
        size = 4, color = "black", fontface = "bold"
      ) +
      geom_text(
        aes(x = 5, y = 25, label = "OWN\nEND ZONE"),
        angle = 90, size = 6, color = "white", fontface = "bold"
      ) +
      geom_text(
        aes(x = 115, y = 25, label = "OPPONENT\nEND ZONE"),
        angle = 90, size = 6, color = "white", fontface = "bold"
      ) +
      labs(
        title = paste("Measuring Defensive Success by Down and Field Position (20 yard chunks)", input$team),
        x = "Yard Line",
        y = "Field Width (yards)"
      ) +
      theme_minimal()
  })
  
  # Yards Gained Prediction (side-by-side bars)
  output$yardsGainedPlot = renderPlot({
    req(input$gameId, input$playId)
    
    play_data = full_data |>
      filter(gameId == input$gameId, playId == input$playId)
    
    if (nrow(play_data) == 0) {
      plot.new()
      title(main = "No data for the selected play.")
      return(NULL)
    }
    
    # Factor alignment helper to match training levels.
    safe_factor = function(x, ref_levels) {
      if (length(ref_levels) > 1) {
        factor(x, levels = ref_levels)
      } else {
        as.character(x)
      }
    }
    if ("down" %in% names(train_data_levels)) {
      play_data$down = safe_factor(play_data$down, train_data_levels$down)
    }
    if ("pff_passCoverage" %in% names(train_data_levels)) {
      play_data$pff_passCoverage = safe_factor(play_data$pff_passCoverage, train_data_levels$pff_passCoverage)
    }
    
    # Create model matrix, predict, and do a simple bar chart of actual vs predicted yards
    fm_yds = model.matrix(
      yardsGained ~ down + yardsToGo + avg_defender_speed + avg_defender_acceleration +
        avg_defender_direction + defender_spacing + total_defensive_motion +
        avg_defender_position_x + avg_offensive_speed + avg_offensive_acceleration +
        avg_offensive_direction + offensive_spacing + total_offensive_motion +
        avg_offensive_position_x + avg_rel_acceleration + avg_rel_direction +
        spacing_diff + total_motion_diff + score_differential + time_pressure +
        pff_passCoverage,
      data = play_data
    )[, -1, drop = FALSE]
    
    pred_yds_play = predict(xgb_model_yards, xgb.DMatrix(fm_yds))
    
    bar_data = data.frame(
      Type  = c("Actual", "Predicted"),
      Yards = c(play_data$yardsGained[1], pred_yds_play[1])
    )
    
    ggplot(bar_data, aes(x = Type, y = Yards, fill = Type)) +
      geom_col(width = 0.5) +
      geom_text(aes(label = round(Yards,2)), vjust = -0.5, size = 5) +
      labs(
        title = paste("Yards Gained for Play", input$playId),
        x = "",
        y = "Yards Gained"
      ) +
      scale_fill_manual(values = c("Actual"="blue","Predicted"="orange")) +
      theme_minimal() +
      theme(legend.position = "none")
  })
  
  # Touchdown Probability
  output$touchdownProbabilityPlot = renderPlotly({
    req(input$gameId, input$playId)
    
    # Retrieve single-play data.
    play_data = full_data |>
      filter(gameId == input$gameId, playId == input$playId)
    
    if (nrow(play_data) == 0) {
      return(
        plot_ly() |>
          layout(title="No data available for the selected play.")
      )
    }
    
    # Align factors with training data.
    safe_factor = function(x, ref_levels) {
      if (length(ref_levels) > 1) {
        factor(x, levels = ref_levels)
      } else {
        as.character(x)
      }
    }
    if ("down" %in% names(train_data_levels)) {
      play_data$down = safe_factor(play_data$down, train_data_levels$down)
    }
    if ("pff_passCoverage" %in% names(train_data_levels)) {
      play_data$pff_passCoverage = safe_factor(play_data$pff_passCoverage, train_data_levels$pff_passCoverage)
    }
    
    fm_td = model.matrix(
      touchdown ~ down + yardsToGo + avg_defender_speed + avg_defender_acceleration +
        avg_defender_direction + defender_spacing + total_defensive_motion +
        avg_defender_position_x + avg_offensive_speed + avg_offensive_acceleration +
        avg_offensive_direction + offensive_spacing + total_offensive_motion +
        avg_offensive_position_x + avg_rel_acceleration + avg_rel_direction +
        spacing_diff + total_motion_diff + score_differential + time_pressure +
        pff_passCoverage,
      data = play_data
    )[, -1, drop = FALSE]
    
    td_pred = predict(xgb_model_td, xgb.DMatrix(fm_td))
    td_prob = round(td_pred[1], 3)
    plot_ly(
      domain = list(x = c(0, 1), y = c(0, 1)),
      value  = td_prob,
      type   = "indicator",
      mode   = "gauge+number",
      gauge  = list(
        axis = list(range = c(0,1)),
        bar  = list(color="green"),
        steps = list(
          list(range=c(0,0.2), color="#e5f5e0"),
          list(range=c(0.2,0.4), color="#c7e9c0"),
          list(range=c(0.4,0.6), color="#a1d99b"),
          list(range=c(0.6,0.8), color="#74c476"),
          list(range=c(0.8,1.0), color="#31a354")
        )
      )
    ) |>
      layout(
        title = list(
          text = paste(
            "Touchdown Probability for Selected Play", input$playId, "<br>",
            paste0(td_prob*100, "%")
          )
        ),
        margin = list(l = 20, r = 20)
      )
  })
}

shinyApp(ui = ui, server = server)