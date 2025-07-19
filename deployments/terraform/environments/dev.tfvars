# Development Environment Configuration
project_name = "mosdac-ai"
environment  = "dev"
aws_region   = "us-east-1"
domain_name  = "dev.mosdac-ai.com"

# VPC Configuration
vpc_cidr         = "10.2.0.0/16"
private_subnets  = ["10.2.1.0/24", "10.2.2.0/24"]
public_subnets   = ["10.2.101.0/24", "10.2.102.0/24"]
intra_subnets    = ["10.2.51.0/24", "10.2.52.0/24"]
single_nat_gateway = true

# EKS Configuration
kubernetes_version = "1.28"
node_instance_types = ["t3.medium"]

node_group_min_size     = 1
node_group_max_size     = 5
node_group_desired_size = 2

spot_instance_types = ["t3.medium", "t3.large"]
spot_min_size       = 0
spot_max_size       = 3
spot_desired_size   = 1

# Database Configuration
postgres_version = "15.4"
db_instance_class = "db.t3.micro"
db_allocated_storage = 20
db_max_allocated_storage = 100
db_backup_retention_period = 1

# Redis Configuration
redis_node_type = "cache.t3.micro"
redis_num_cache_clusters = 1

# Additional Tags
additional_tags = {
  CostCenter   = "Engineering"
  Owner        = "GravitasOps"
  Environment  = "Development"
  AutoShutdown = "Enabled"
}
