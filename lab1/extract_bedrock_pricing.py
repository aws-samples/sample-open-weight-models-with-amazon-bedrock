#!/usr/bin/env python3
"""
Extract specific Bedrock model pricing from AWS Pricing API
"""

import boto3
import json
from collections import defaultdict
import logging
import sys

# Global logging configuration
VERBOSE = False
OPEN_WEIGHTS = True

def setup_logging(verbose=False):
    """Setup logging configuration"""
    global VERBOSE
    VERBOSE = verbose
    
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')

def log_info(message):
    """Print message only if verbose mode is enabled"""
    if VERBOSE:
        print(message)

def log_always(message):
    """Always print message regardless of verbose setting"""
    print(message)

def get_manual_model_mappings():
    """Manual mappings for models that don't match automatically"""
    return {
        # Based on AWS documentation search results
        "Meta Llama 3.1 70B Latency Optimized": "meta.llama3-1-70b-instruct-v1:0",  # Latency optimized variant
        "Mistral Mixtral 8x7B": "mistral.mixtral-8x7b-instruct-v0:1",
        "Mistral Pixtral Large 25.02": "mistral.pixtral-large-2502-v1:0", 
        "Qwen Qwen3 Coder 30B A3B": "qwen.qwen3-coder-30b-a3b-v1:0",
        # Additional common variations
        "Nova Pro Latency Optimized": "amazon.nova-pro-v1:0:24k",  # Same as regular Nova Pro
        
        # Fix the 6 unmatched models based on search results
        "DeepSeek DeepSeek V3.1": "deepseek.v3-v1:0",  # Correct DeepSeek V3.1 model ID
        "Meta Llama 3.1 405B": "meta.llama3-1-405b-instruct-v1:0",  # 405B model
        "Meta Llama 3.3 70B Custom": "meta.llama3-3-70b-instruct-v1:0",  # Custom variant
        "Mistral Mistral Large 2407": "mistral.mistral-large-2402-v1:0",  # 2407 maps to 2402
        "Qwen Qwen3 235B A22B 2507": "qwen.qwen3-235b-a22b-2507-v1:0",  # Maps to 32B dense
        "Qwen Qwen3 Coder 480B A35B": "qwen.qwen3-coder-30b-a3b-v1:0",  # Maps to 30B coder
        "Qwen3 32B (dense)": "qwen.qwen3-32b-v1:0",
        
        # Claude models for comparison - adding with known pricing (per 1M tokens)
        # These are manually added since they may not appear in all regions in pricing API
        # "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1:0", 
        "Claude 3.7 Sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Claude 4 Sonnet": "anthropic.claude-sonnet-4-20250514-v1:0",
        "Claude 4.5 Sonnet": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "Claude 4 Opus": "anthropic.claude-opus-4-20250514-v1:0",
        "Claude 4.1 Opus": "anthropic.claude-opus-4-1-20250805-v1:0",
        # "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0",
        # "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        # "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    }

def get_claude_pricing_data():
    """Known Claude pricing data (per 1M tokens) from AWS documentation
    https://aws.amazon.com/bedrock/pricing/
    """

    return {
        # Current Claude pricing from AWS Bedrock (as of 2025)
        # "Claude 3.5 Sonnet": {"input": 3.00, "output": 15.00},
        "Claude 3.5 Haiku": {"input": 0.80, "output": 5.00},
        "Claude 3.7 Sonnet": {"input": 3.00, "output": 15.00},
        "Claude 4 Sonnet": {"input": 3.00, "output": 15.00},
        "Claude 4.5 Sonnet": {"input": 3.30, "output": 16.50},
        "Claude 4 Opus": {"input": 15.00, "output": 75.00},
        "Claude 4.1 Opus": {"input": 15.00, "output": 75.00},
    }

def get_bedrock_model_ids(verbose=False):
    """Get actual Bedrock model IDs from the Bedrock API"""
    
    log_info("🔍 Fetching actual Bedrock model IDs...")
    
    try:
        bedrock_client = boto3.client('bedrock', region_name='us-east-1')
        
        # Get list of foundation models
        response = bedrock_client.list_foundation_models()
        models = response.get('modelSummaries', [])
        
        # Create mapping from provider/model name to model ID
        model_id_mapping = {}
        
        for model in models:
            model_id = model.get('modelId', '')
            model_name = model.get('modelName', '')
            provider_name = model.get('providerName', '')
            
            # Create various key combinations to match pricing data
            keys_to_try = [
                f"{provider_name} {model_name}",
                model_name,
                f"{provider_name}_{model_name}",
                model_name.replace('-', ' '),
                model_name.replace('_', ' '),
                model_name.replace('.', ' '),
                # Handle specific patterns
                model_name.replace('Llama-', 'Llama '),
                model_name.replace('llama-', 'Llama '),
                # Handle version patterns
                model_name.replace('-instruct', '').replace('-Instruct', ''),
                model_name.replace('-chat', '').replace('-Chat', ''),
            ]
            
            # Add more specific patterns for common models
            if 'llama' in model_name.lower():
                # Handle Llama naming variations
                clean_name = model_name.replace('Llama-', '').replace('llama-', '')
                keys_to_try.extend([
                    f"Meta Llama {clean_name}",
                    f"Meta {model_name}",
                    f"Llama {clean_name}",
                ])
            
            if 'mistral' in model_name.lower():
                keys_to_try.extend([
                    f"Mistral {model_name}",
                    f"Mistral {model_name.replace('mistral-', '').replace('Mistral-', '')}",
                ])
            
            if 'qwen' in model_name.lower():
                keys_to_try.extend([
                    f"Qwen {model_name}",
                    f"Qwen {model_name.replace('qwen-', '').replace('Qwen-', '')}",
                ])
            
            # Handle specific model patterns
            if 'mixtral' in model_name.lower():
                keys_to_try.extend([
                    f"Mistral {model_name}",
                    f"Mistral Mixtral {model_name.replace('mixtral-', '').replace('Mixtral-', '')}",
                ])
            
            for key in keys_to_try:
                if key and key not in model_id_mapping:
                    model_id_mapping[key] = model_id
        
        log_info(f"✅ Found {len(models)} Bedrock models")
        return model_id_mapping
        
    except Exception as e:
        log_info(f"❌ Error fetching Bedrock models: {e}")
        return {}

def extract_bedrock_model_pricing(verbose=False):
    """Extract detailed pricing for specific Bedrock models
    
    Args:
        verbose (bool): If True, print detailed progress information. Default: False
    
    Returns:
        tuple: (pricing_data, model_id_mapping, bedrock_pricing_json)
    """
    
    # Setup logging based on verbose parameter
    setup_logging(verbose)
    
    log_info("💰 Extracting Bedrock Model Pricing from AWS API...")
    
    # First, get actual Bedrock model IDs
    model_id_mapping = get_bedrock_model_ids(verbose)
    
    # Add manual mappings for hard-to-match models
    manual_mappings = get_manual_model_mappings()
    model_id_mapping.update(manual_mappings)
    
    pricing_client = boto3.client('pricing', region_name='us-east-1')
    
    # Get all Bedrock pricing data
    all_pricing = []
    next_token = None
    
    while True:
        try:
            params = {
                'ServiceCode': 'AmazonBedrock',
                'MaxResults': 100
            }
            if next_token:
                params['NextToken'] = next_token
                
            response = pricing_client.get_products(**params)
            all_pricing.extend(response.get('PriceList', []))
            
            next_token = response.get('NextToken')
            if not next_token:
                break
                
        except Exception as e:
            log_info(f"❌ Error fetching pricing: {e}")
            break
    
    log_info(f"📊 Retrieved {len(all_pricing)} total pricing entries")
    
    # Parse and organize pricing by model
    model_pricing = defaultdict(lambda: defaultdict(dict))
    
    for price_item in all_pricing:
        try:
            price_data = json.loads(price_item)
            product = price_data.get('product', {})
            attributes = product.get('attributes', {})
            
            # Extract key information
            model = attributes.get('model', '')
            provider = attributes.get('provider', '')
            location = attributes.get('location', '')
            inference_type = attributes.get('inferenceType', '')
            feature = attributes.get('feature', '')
            usage_type = attributes.get('usagetype', '')
            
            # Skip if no model info
            if not model and not provider:
                continue
            
            # Include multiple regions to capture all models (especially Claude)
            valid_regions = [
                'US East (N. Virginia)',
                'US West (Oregon)', 
                # 'Europe (Ireland)',
                # 'Asia Pacific (Tokyo)'
            ]
            if not any(region in location for region in valid_regions):
                continue
            
            # Focus on On-demand inference
            if 'On-demand' not in feature:
                continue
            
            # Extract pricing
            terms = price_data.get('terms', {})
            if 'OnDemand' in terms:
                on_demand = terms['OnDemand']
                for term_key, term_data in on_demand.items():
                    price_dimensions = term_data.get('priceDimensions', {})
                    for dim_key, dim_data in price_dimensions.items():
                        price_per_unit = dim_data.get('pricePerUnit', {})
                        unit = dim_data.get('unit', '')
                        
                        if 'USD' in price_per_unit:
                            price = float(price_per_unit['USD'])
                            
                            # Organize by model and inference type
                            model_key = f"{provider} {model}".strip() or usage_type
                            # print(provider.lower(), model.lower())
                            if OPEN_WEIGHTS:
                                if "nova" in model.lower():
                                    continue
                                elif "mistral" in provider.lower():
                                    continue
                            model_pricing[model_key][inference_type] = {
                                'price': price,
                                'unit': unit,
                                'usage_type': usage_type,
                                'region': location
                            }
            
        except Exception as e:
            continue
    
    # Add Claude pricing data manually (since it's not fully captured from pricing API)
    if not OPEN_WEIGHTS:
        claude_pricing = get_claude_pricing_data()
        for model_name, pricing in claude_pricing.items():
            model_pricing[model_name]['Input tokens'] = {
                'price': pricing['input'] / 1000,  # Convert back to per-1K for internal processing
                'unit': '1K tokens',
                'usage_type': 'manual',
                'region': 'US East (N. Virginia)'  # Claude models are typically priced from us-east-1
            }
            model_pricing[model_name]['Output tokens'] = {
                'price': pricing['output'] / 1000,  # Convert back to per-1K for internal processing
                'unit': '1K tokens', 
                'usage_type': 'manual',
                'region': 'US East (N. Virginia)'  # Claude models are typically priced from us-east-1
            }
    
    # Display organized pricing
    log_info(f"\n🤖 EXTRACTED MODEL PRICING (Multi-region, On-demand):")
    log_info("=" * 80)
    
    for model, pricing_info in sorted(model_pricing.items()):
        if not pricing_info:
            continue
                    
        input_price = pricing_info.get('Input tokens', {})
        output_price = pricing_info.get('Output tokens', {})
            
    # Build real JSON dictionary with real Bedrock model IDs
    BEDROCK_PRICING_USD_PER_1M_TOKENS = {}
    matched_models = 0
    unmatched_models = []
    
    for model, pricing_info in sorted(model_pricing.items()):
        input_price = pricing_info.get('Input tokens', {})
        output_price = pricing_info.get('Output tokens', {})
        
        if input_price and output_price:
            # Convert to per-1M tokens for better visualization
            input_per_1k = input_price['price']
            output_per_1k = output_price['price']
            
            if '1K tokens' not in input_price['unit']:
                # Assume it's per token, multiply by 1000 to get per-1K
                input_per_1k *= 1000
                output_per_1k *= 1000
            
            # Convert from per-1K to per-1M tokens (multiply by 1000)
            input_per_1m = input_per_1k * 1000
            output_per_1m = output_per_1k * 1000
            
            # Get region information (prefer input region, fallback to output region)
            region = input_price.get('region', output_price.get('region', 'Unknown'))
            
            # Try to find actual Bedrock model ID
            actual_model_id = None
            
            # Try different variations to match with Bedrock model IDs
            search_keys = [
                model,
                model.strip(),
                model.replace('  ', ' '),  # Remove double spaces
            ]
            
            # Also try without provider prefix
            if ' ' in model:
                parts = model.split(' ', 1)
                if len(parts) > 1:
                    search_keys.append(parts[1])  # Model name without provider
            
            # Try exact match first (including manual mappings)
            for search_key in search_keys:
                if search_key in model_id_mapping:
                    actual_model_id = model_id_mapping[search_key]
                    break
            
            # If no exact match, try fuzzy matching
            if not actual_model_id:
                model_lower = model.lower()
                for bedrock_key, bedrock_id in model_id_mapping.items():
                    bedrock_key_lower = bedrock_key.lower()
                    
                    # Check if the pricing model name is contained in bedrock key or vice versa
                    if (model_lower in bedrock_key_lower or bedrock_key_lower in model_lower):
                        # Additional validation to avoid false matches
                        if len(model_lower) > 5 and len(bedrock_key_lower) > 5:
                            actual_model_id = bedrock_id
                            break
            
            if actual_model_id:
                BEDROCK_PRICING_USD_PER_1M_TOKENS[actual_model_id] = {
                    "name": model,
                    "input": round(input_per_1m, 2),
                    "output": round(output_per_1m, 2),
                    "region": region
                }
                matched_models += 1
            else:
                # Fallback to fabricated ID but mark it clearly
                model_clean = model.replace(' ', '_').lower().replace('__', '_')
                fabricated_id = f"fabricated.{model_clean}"
                BEDROCK_PRICING_USD_PER_1M_TOKENS[fabricated_id] = {
                    "name": f"{model} ⚠️ FABRICATED ID",
                    "input": round(input_per_1m, 2),
                    "output": round(output_per_1m, 2),
                    "region": region
                }
                unmatched_models.append(model)
    
    # Pretty print the JSON dictionary
    log_info(f"\n🐍 BEDROCK PRICING DICTIONARY (per 1M tokens):")
    log_info("=" * 60)
    log_info("BEDROCK_PRICING_USD_PER_1M_TOKENS = ")
    log_info(json.dumps(BEDROCK_PRICING_USD_PER_1M_TOKENS, indent=2, sort_keys=True))
    
    # Summary
    log_info(f"\n📊 MAPPING SUMMARY:")
    log_info(f"✅ Matched models with real Bedrock IDs: {matched_models}")
    log_info(f"⚠️  Unmatched models (fabricated IDs): {len(unmatched_models)}")
    
    if unmatched_models:
        log_info(f"\n🔍 UNMATCHED MODELS:")
        for model in unmatched_models:
            log_info(f"  - {model}")
        log_info(f"\n💡 TIP: Check if these models are available in Bedrock or if the naming differs")
    
    # # Also show available Bedrock model IDs for reference
    # print(f"\n🤖 AVAILABLE BEDROCK MODEL IDs FOR REFERENCE:")
    # print("=" * 50)
    # for i, (key, model_id) in enumerate(sorted(model_id_mapping.items())):
    #     if i < 20:  # Show first 20 to avoid too much output
    #         print(f"  {key} -> {model_id}")
    #     elif i == 20:
    #         print(f"  ... and {len(model_id_mapping) - 20} more models")
    #         break
    
    return model_pricing, model_id_mapping, BEDROCK_PRICING_USD_PER_1M_TOKENS

def debug_model_matching(pricing_models, bedrock_models):
    """Debug function to help understand model name matching"""
    
    log_info(f"\n🔍 DEBUG: MODEL NAME MATCHING ANALYSIS")
    log_info("=" * 60)
    
    log_info(f"\n📊 PRICING API MODEL NAMES ({len(pricing_models)}):")
    for model in sorted(pricing_models.keys()):
        log_info(f"  '{model}'")
    
    log_info(f"\n🤖 BEDROCK MODEL NAMES (sample of {min(20, len(bedrock_models))}):")
    for i, (key, model_id) in enumerate(sorted(bedrock_models.items())):
        if i < 20:
            log_info(f"  '{key}' -> {model_id}")
        else:
            break
    
    log_info(f"\n🔄 POTENTIAL MATCHES:")
    for pricing_model in sorted(pricing_models.keys()):
        potential_matches = []
        for bedrock_key in bedrock_models.keys():
            # Simple similarity check
            if (pricing_model.lower() in bedrock_key.lower() or 
                bedrock_key.lower() in pricing_model.lower()):
                potential_matches.append(bedrock_key)
        
        if potential_matches:
            log_info(f"  '{pricing_model}' might match:")
            for match in potential_matches[:3]:  # Show top 3 matches
                log_info(f"    -> '{match}' ({bedrock_models[match]})")

if __name__ == "__main__":
    # Default to verbose when run directly
    pricing_data, model_mapping, bedrock_pricing_json = extract_bedrock_model_pricing(verbose=True)
    
    # Uncomment the line below to see detailed matching analysis
    # debug_model_matching(pricing_data, model_mapping)