#!/usr/bin/env Rscript

# Load crabs dataset (from MASS package)
library(MASS)

# Load the crabs dataset
data(crabs)

cat("=== Crabs Dataset ===\n")
cat("Description: Morphological measurements on Leptograpsus crabs\n")
cat("Samples:", nrow(crabs), "\n")
cat("Features:", ncol(crabs), "\n\n")

cat("Dataset structure:\n")
print(str(crabs))

cat("\nFirst few rows:\n")
print(head(crabs))

cat("\nColumn descriptions:\n")
cat("- sp: species (B=blue, O=orange)\n")
cat("- sex: sex (M=male, F=female)\n")
cat("- index: individual number\n")
cat("- FL: frontal lobe size (mm)\n")
cat("- RW: rear width (mm)\n")
cat("- CL: carapace length (mm)\n")
cat("- CW: carapace width (mm)\n")
cat("- BD: body depth (mm)\n\n")

# Extract numeric features only (columns 4-8: FL, RW, CL, CW, BD)
numeric_features <- crabs[, 4:8]

cat("Numeric features extracted (FL, RW, CL, CW, BD):\n")
print(summary(numeric_features))

# Save the numeric data to CSV
write.csv(numeric_features, "data/crabs_data.csv", row.names = FALSE)
cat("\nSaved numeric features to: crabs_data.csv\n")

# Save metadata (species and sex)
metadata <- crabs[, 1:2]
write.csv(metadata, "crabs_metadata.csv", row.names = FALSE)
cat("Saved metadata to: crabs_metadata.csv\n")

# Create a combined label for visualization
crabs$group <- paste(crabs$sp, crabs$sex, sep = "_")
write.csv(crabs[, c("sp", "sex", "group")], "data/crabs_labels.csv", row.names = FALSE)
cat("Saved labels to: crabs_labels.csv\n")

cat("\nGroups in dataset:\n")
print(table(crabs$group))
