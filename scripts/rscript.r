# install.packages("umap")

# install.packages("reshape2")

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(umap)
library(stats)
library(reshape2)

# Load the dataset (first 10,000 rows)
df <- read.csv("D:/Data_Science/project/nyc_taxi_2019_combined.csv", nrows = 10000)

# Convert datetime columns
df$tpep_pickup_datetime <- as.POSIXct(df$tpep_pickup_datetime)
df$tpep_dropoff_datetime <- as.POSIXct(df$tpep_dropoff_datetime)

# Remove invalid trip distances and fares
df <- df %>% filter(trip_distance > 0 & fare_amount > 0)

# Standardize numeric columns
numeric_cols <- c('trip_distance', 'fare_amount', 'total_amount', 'tolls_amount')
df[numeric_cols] <- scale(df[numeric_cols])

# Convert categorical fields to numerical encoding
df$payment_type <- as.integer(df$payment_type)
df$vendorid <- as.integer(df$vendorid)
df$ratecodeid <- as.integer(df$ratecodeid)

# Remove outliers using IQR
Q1 <- apply(df[, c('fare_amount', 'trip_distance')], 2, function(x) quantile(x, 0.25))
Q3 <- apply(df[, c('fare_amount', 'trip_distance')], 2, function(x) quantile(x, 0.75))
IQR <- Q3 - Q1
df <- df %>% filter(!(fare_amount < (Q1['fare_amount'] - 1.5 * IQR['fare_amount']) | fare_amount > (Q3['fare_amount'] + 1.5 * IQR['fare_amount'])))
df <- df %>% filter(!(trip_distance < (Q1['trip_distance'] - 1.5 * IQR['trip_distance']) | trip_distance > (Q3['trip_distance'] + 1.5 * IQR['trip_distance'])))

cat("Data Cleaning Complete!\n")

# Histogram of fares
print(ggplot(df, aes(x = fare_amount)) +
  geom_histogram(bins = 50, fill = 'blue', alpha = 0.6) +
  geom_density(aes(y = ..count..), color = 'red', size = 1) +
  ggtitle("Fare Amount Distribution") +
  theme_minimal())

readline(prompt = "Press [Enter] to see the next plot...")

# Histogram of trip distances
print(ggplot(df, aes(x = trip_distance)) +
  geom_histogram(bins = 50, fill = 'blue', alpha = 0.6) +
  geom_density(aes(y = ..count..), color = 'red', size = 1) +
  ggtitle("Trip Distance Distribution") +
  theme_minimal())

readline(prompt = "Press [Enter] to see the next plot...")

# Correlation Matrix
cor_matrix <- cor(df[, numeric_cols])
print(ggplot(melt(cor_matrix), aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  ggtitle("Feature Correlations"))

readline(prompt = "Press [Enter] to see the next plot...")

# Apply PCA
pca <- prcomp(df[, c('trip_distance', 'fare_amount')], center = TRUE, scale. = TRUE)
pca_data <- data.frame(pca$x)
print(ggplot(pca_data, aes(x = PC1, y = PC2)) +
  geom_point() +
  ggtitle("PCA Representation") +
  theme_minimal())

readline(prompt = "Press [Enter] to see the next plot...")

# Apply UMAP
umap_data <- umap(df[, c('trip_distance', 'fare_amount')])
umap_df <- as.data.frame(umap_data$layout)
print(ggplot(umap_df, aes(x = V1, y = V2)) +
  geom_point() +
  ggtitle("UMAP Representation") +
  theme_minimal())

# Extract time-based features
df$pickup_hour <- as.integer(format(df$tpep_pickup_datetime, "%H"))
df$pickup_day <- as.integer(format(df$tpep_pickup_datetime, "%d"))
df$pickup_weekday <- as.integer(format(df$tpep_pickup_datetime, "%u"))
df$pickup_month <- as.integer(format(df$tpep_pickup_datetime, "%m"))

# Trip duration (minutes)
df$trip_duration <- as.numeric(difftime(df$tpep_dropoff_datetime, df$tpep_pickup_datetime, units = "mins"))

# Speed (miles per hour)
df$speed_mph <- df$trip_distance / (df$trip_duration / 60 + 1e-6)

# Weekend indicator
df$is_weekend <- ifelse(df$pickup_weekday >= 6, 1, 0)

# Rush hour indicator (Morning: 7-9 AM, Evening: 5-7 PM)
df$is_rush_hour <- ifelse(df$pickup_hour %in% c(7, 8, 17, 18), 1, 0)

# Nighttime trip indicator (Midnight - 5 AM)
df$is_night <- ifelse(df$pickup_hour < 6, 1, 0)

# High-fare trip classification
df$high_fare <- ifelse(df$fare_amount > median(df$fare_amount, na.rm = TRUE), 1, 0)

# Perform ANOVA test
anova_result <- aov(fare_amount ~ factor(payment_type), data = df)
summary(anova_result)