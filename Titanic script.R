install.packages("randomForest")
install.packages("ggplot")
install.packages("tidyr")
library(tidyverse)
library(dplyr)
library(randomForest)
library(ggplot2)
library(tidyr)

df <- read.csv("tested.csv")

glimpse(df)

# Check why there are many missing values for Cabin field
third_class_cabins <- df[df$Pclass == 3, "Cabin"]
print(unique(third_class_cabins))
third_class_cabins

# Check unique cabin values
unique(df$Cabin)

#Check the cabin assignment for first and second class
first_class_cabins <- df[df$Pclass == 1, "Cabin"]
print(unique(first_class_cabins))

second_class_cabins <- df[df$Pclass == 2, "Cabin"]
print(unique(second_class_cabins))

glimpse(df)

# Replace white spaces in the Cabin column with "None"
df$Cabin <- ifelse(grepl("^\\s*$", df$Cabin), "None", df$Cabin)

print(df)

# Check summary of whole dataset
summary(df)

# Noticed we have one NA in Fare 
# Get the average and do not include NA in the computation using na.rm
mean_fare <- mean(df$Fare, na.rm = TRUE)
mean_fare

df$Fare <- replace_na(df$Fare, 35.63)

# Noticed that we have NAs in age, need to fix that
missing_age_data <- df %>% filter(is.na(Age))
complete_age_data <- df %>% filter(!is.na(Age))

summary(complete_age_data)

# Rename the Embarked locations
df <- df %>% 
  mutate(Embarked = case_when(
    Embarked == "C" ~ "Cherbourg",
    Embarked == "S" ~ "Southampton",
    Embarked == "Q" ~ "Queenstown",
    TRUE ~ Embarked  # Keep other values unchanged
  ))

# Model building section to fill missing Age values
# Features and target variable
features <- c("Pclass", "SibSp", "Parch", "Fare")
target <- "Age"

# Train a random forest model on complete data
rf_model <- randomForest(
  formula = Age ~ .,
  data = complete_age_data[, c(features, target)],
  ntree = 100
)

# Predict missing Age values using the model
missing_age_data$Age <- predict(rf_model, newdata = missing_age_data)

# Create a new data frame for imputed Age values
imputed_data <- bind_rows(missing_age_data, complete_age_data)

# Print the first few rows of imputed data
print(head(imputed_data))

summary(imputed_data)

# Combine the imputed data with original data frame (has missing values)
df <- left_join(df, imputed_data %>% select(PassengerId, Age), by = "PassengerId")

# Drop Age.x
df$Age.x <- NULL

# Replace Age.y
df$Age <- df$Age.y
df$Age.y <- NULL

summary(df)

# End of model building for filling missing values

# Check if all PassengerId is unique
id_unique <- all(length(unique(df$PassengerId)) == length(df$PassengerId))

if (id_unique) {
  print("All values in the column are unique.")
} else {
  print("There are duplicate values in the column.")
}

# Check if all Names are unique
name_unique <- all(length(unique(df$Name)) == length(df$Name))
unique(name_unique)
?unique

# Check if Fare is unique (should not be)
fare_unique <- all(length(unique(df$Fare)) == length(df$Fare))
unique(fare_unique)

# Check if sex values are unique
sex_unique <- unique(df$Sex)
print(sex_unique)

save.image(file = "Titanic.RData")

# Data visualizations
# Create the data for survived and fatalities
survivor_counts <- df %>%
  group_by(Pclass, Survived) %>%
  summarise(Count = n())

# Number of survivors and fatalities based on passenger class 
ggplot(survivor_counts, aes(x = factor(Pclass), y = Count, fill = factor(Survived))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Survivors and Fatalities by Passenger Class", x = "Passenger Class", y = "Count") +
  scale_fill_manual(values = c("0" = "#CCCCCC", "1" = "#4477AA")) +  # Specify fill colors for "died" and "survived"
  theme_minimal()

# average age of those who survived and died
ggplot(df, aes(x = Age, fill = factor(Survived))) +
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Histogram of Age for Survivors and Fatalities", x = "Age", y = "Count") +
  scale_fill_manual(values = c("0" = "#CCCCCC", "1" = "#4477AA")) +  # Specify fill colors for "died" and "survived"
  theme_minimal()

save.image(file = "Titanic.RData")

# Average fare per class
average_fare <- df %>%
  group_by(Pclass) %>%
  summarise(average_fare = mean(Fare, na.rm = TRUE))

# Visualize the average fare per class
ggplot(average_fare, aes(x = factor(Pclass), y = average_fare, fill = factor(Pclass))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Average Fare per Class", x = "Passenger Class", y = "Average Fare") +
  scale_fill_manual(values = c("1" = "#4477AA", "2" = "#88CCEE", "3" = "#CC6677")) +  # Specify fill colors
  theme_minimal()

# Visualize the average fare per Embarked locations
# Average fare per embarked locations
average_embarked <- df %>%
  group_by(Embarked) %>%
  summarise(average_fare = mean(Fare), na.rm = TRUE)

# Count of passengers per embarked locations
passengers_per_location <- df %>%
  group_by(Embarked) %>%
  summarise(Passengers = n_distinct(PassengerId))

# Merge the last two data frame 
average_embarked_count_passengers <- merge(average_embarked, passengers_per_location, by = "Embarked")

# Create a bubble chart for the merged data
ggplot(average_embarked_count_passengers, aes(x = Embarked)) +
  geom_line(aes(y = average_fare, fill = "Average Fare"), stat = "identity") +
  geom_bar(aes(y = Passengers, fill = "Passengers"), stat = "identity") +
  labs(title = "Comparison of Average Fare and Passengers per Embarked Location", x = "Embarked", y = "Value") +
  scale_fill_manual(values = c("Average Fare" = "#4477AA", "Passengers" = "#88CCEE")) +
  theme_minimal() +
  theme(legend.position = "top")
