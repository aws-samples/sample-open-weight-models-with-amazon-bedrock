#!/usr/bin/env python3
"""
Minimal LLM comparison tool for Jupyter notebooks
Self-contained version optimized for direct model ID usage
"""

import boto3
import pandas as pd
from IPython.display import display
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Global pricing data
_external_pricing_data = None

class LLMCall:
    """LLM Call class for tracking model execution state"""
    def __init__(self, name, model_id):
        self.name = name.strip() if name else name
        self.model_id = model_id.strip() if model_id else model_id
        self.status = "Waiting..."
        self.response = ""
        self.latency = 0
        self.ttft = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0
        self.pricing_info = None
        self.successful_region = None
        self.tokens_per_word = 0
        self.word_count = 0
        self.throughput_tokens_per_sec = 0
        self.throughput_words_per_sec = 0

def set_pricing_data(bedrock_pricing_json):
    """Set external pricing data to avoid recalculation"""
    global _external_pricing_data
    _external_pricing_data = bedrock_pricing_json
    print(f"✅ Using provided pricing data for {len(bedrock_pricing_json)} models")

def find_model_pricing(model_id, pricing_data):
    """Find pricing with simple fallback logic"""
    if model_id in pricing_data:
        return pricing_data[model_id]
    
    # Try without region prefix
    for prefix in ["us.", "eu.", "ap."]:
        if model_id.startswith(prefix):
            clean_id = model_id[len(prefix):]
            if clean_id in pricing_data:
                return pricing_data[clean_id]
    
    return None

def calculate_model_cost_simple(model_id, input_tokens, output_tokens, pricing_data=None):
    """Calculate cost using provided pricing data or fallback to estimates"""
    
    if pricing_data is None:
        pricing_data = _external_pricing_data or {}
    
    model_pricing = find_model_pricing(model_id, pricing_data)
    
    if model_pricing:
        # Pricing is per 1M tokens, convert to actual usage
        input_cost = (input_tokens / 1_000_000) * model_pricing['input']
        output_cost = (output_tokens / 1_000_000) * model_pricing['output']
        total_cost = input_cost + output_cost
        return total_cost, model_pricing
    else:
        # Fallback to simple estimation
        fallback_cost = (input_tokens + output_tokens) * 0.0001
        return fallback_cost, None

def parse_model_input(model_input):
    """Parse model input - simplified for direct model IDs"""
    if isinstance(model_input, dict):
        # Clean model_id in dict if present
        if 'model_id' in model_input:
            model_input['model_id'] = model_input['model_id'].strip()
        return model_input
    
    # Clean whitespace and tabs from model ID string
    clean_model_id = str(model_input).strip()
    
    # Generate friendly name from model ID
    name = clean_model_id.split('.')[-1].replace('-', ' ').title()
    return {"name": name, "model_id": clean_model_id}

def get_region_display(llm_call):
    """Get region display string for results"""
    if hasattr(llm_call, 'successful_region') and llm_call.successful_region:
        region = llm_call.successful_region
        return region + (' (fallback)' if region != 'us-east-1' else ' (default)')
    elif any(llm_call.model_id.startswith(prefix) for prefix in ["us.", "eu.", "ap."]):
        return f"{llm_call.model_id.split('.')[0].upper()} prefix"
    return 'us-east-1 (default)'

def calculate_metrics(llm_call):
    """Calculate tokens per word and throughput metrics"""
    if llm_call.response:
        llm_call.word_count = len(llm_call.response.split())
        llm_call.tokens_per_word = llm_call.output_tokens / llm_call.word_count if llm_call.word_count > 0 else 0
    else:
        llm_call.word_count = 0
        llm_call.tokens_per_word = 0
    
    if llm_call.latency > 0:
        llm_call.throughput_tokens_per_sec = llm_call.output_tokens / llm_call.latency
        llm_call.throughput_words_per_sec = llm_call.word_count / llm_call.latency
    else:
        llm_call.throughput_tokens_per_sec = 0
        llm_call.throughput_words_per_sec = 0

def _bedrock_call_with_timeout(client, model_id, prompt, timeout_seconds=30):
    """Execute Bedrock call with timeout"""
    response = client.converse_stream(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.7}
    )
    return response

def call_bedrock_sync(llm_call, prompt, pricing_data=None, timeout_seconds=30):
    """Synchronous call to Bedrock with region fallback and cross-region retry"""
    
    regions_to_try = ['us-east-1', 'us-west-2']
    
    llm_call.status = "Calling..."
    first_token = True
    
    # List of model IDs to try (original + cross-region prefixes)
    model_ids_to_try = [llm_call.model_id]
    if not any(llm_call.model_id.startswith(prefix) for prefix in ["us.", "eu.", "ap."]):
        model_ids_to_try.extend([f"us.{llm_call.model_id}", f"eu.{llm_call.model_id}", f"ap.{llm_call.model_id}"])
    
    last_error = None
    
    # Try each region
    for region in regions_to_try:
        client = boto3.client('bedrock-runtime', region_name=region)
        
        # Try each model ID variant in this region
        for attempt, model_id in enumerate(model_ids_to_try):
            try:
                if region != 'us-east-1' or attempt > 0:
                    print(f"  🔄 Trying region {region} with model ID: {model_id}")

                start_time = time.time()
                
                # Execute Bedrock call with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_bedrock_call_with_timeout, client, model_id, prompt, timeout_seconds)
                    try:
                        response = future.result(timeout=timeout_seconds)
                    except TimeoutError:
                        raise Exception(f"Request timed out after {timeout_seconds} seconds")
                
                # Success - update model info
                if region != 'us-east-1' or attempt > 0:
                    print(f"  ✅ Success in region {region} with model ID: {model_id}")
                
                llm_call.model_id = model_id
                llm_call.successful_region = region
                llm_call.status = "Streaming..."
                
                for event in response['stream']:
                    if 'contentBlockDelta' in event:
                        delta = event['contentBlockDelta']['delta']
                        if 'text' in delta:
                            if first_token:
                                llm_call.ttft = time.time() - start_time
                                first_token = False
                            llm_call.response += delta['text']
                    
                    elif 'messageStop' in event:
                        llm_call.latency = time.time() - start_time
                        llm_call.status = "Complete"
                    
                    elif 'metadata' in event:
                        usage = event['metadata'].get('usage', {})
                        llm_call.input_tokens = usage.get('inputTokens', 100)
                        llm_call.output_tokens = usage.get('outputTokens', len(llm_call.response.split()))
                        
                        # Calculate metrics
                        calculate_metrics(llm_call)
                        
                        # Calculate cost using original model_id for pricing lookup
                        original_model_id = model_ids_to_try[0]
                        llm_call.cost, llm_call.pricing_info = calculate_model_cost_simple(
                            original_model_id, llm_call.input_tokens, llm_call.output_tokens, pricing_data)
                
                return  # Success
                
            except Exception as e:
                last_error = e
                if region == 'us-east-1' and attempt == 0:
                    print(f"  ❌ Failed in {region} with original model ID: {model_id}")
                    print(f"     Error: {str(e)[:100]}...")
                elif attempt < len(model_ids_to_try) - 1:
                    print(f"  ❌ Failed in {region} with model ID: {model_id}")
                continue
    
    # All attempts failed
    llm_call.status = "Error"
    llm_call.response = f"Error (tried {len(regions_to_try)} regions with {len(model_ids_to_try)} model ID variants each): {str(last_error)}"
    llm_call.latency = time.time() - start_time
    llm_call.successful_region = None
    print(f"  ❌ All region and model ID attempts failed for {llm_call.name}")

def wrap_text_by_words(text, words_per_line=20):
    """Wrap text to next line every N words"""
    if not text:
        return text
    
    words = text.split()
    lines = []
    
    for i in range(0, len(words), words_per_line):
        line = ' '.join(words[i:i + words_per_line])
        lines.append(line)
    
    return '\n'.join(lines)

def display_model_result(llm_call, index, total):
    """Display individual model result"""
    
    status_icon = "✅" if llm_call.status == "Complete" else "❌" if llm_call.status == "Error" else "⏳"
    
    print(f"\n{status_icon} Model {index}/{total}: {llm_call.name}")
    print("─" * 60)
    print(f"Status:        {llm_call.status}")
    print(f"Model ID:      {llm_call.model_id}")
    print(f"Region:        {get_region_display(llm_call)}")
    print(f"Latency:       {llm_call.latency:.2f}s")
    print(f"TTFT:          {llm_call.ttft:.2f}s")
    print(f"Input Tokens:  {llm_call.input_tokens}")
    print(f"Output Tokens: {llm_call.output_tokens}")
    
    # Tokens per word
    if llm_call.response and llm_call.output_tokens > 0:
        print(f"Tokens/Word:   {llm_call.tokens_per_word:.2f} ({llm_call.word_count} words)")
    else:
        print(f"Tokens/Word:   N/A")
    
    # Throughput
    if llm_call.latency > 0 and llm_call.output_tokens > 0:
        print(f"Throughput:    {llm_call.throughput_tokens_per_sec:.1f} tokens/sec, {llm_call.throughput_words_per_sec:.1f} words/sec")
    else:
        print(f"Throughput:    N/A")
    
    # Cost
    cost_cents = llm_call.cost * 100
    if llm_call.pricing_info:
        print(f"Cost:          {cost_cents:.4f}¢ (Real pricing)")
        print(f"Input Rate:    ${llm_call.pricing_info['input']}/1M tokens")
        print(f"Output Rate:   ${llm_call.pricing_info['output']}/1M tokens")
    else:
        print(f"Cost:          {cost_cents:.4f}¢ (Estimated)")
    
    # Response
    print(f"\n📝 Response:")
    print("─" * 40)
    if llm_call.status == "Error":
        print(f"❌ ERROR: {llm_call.response}")
    else:
        # Wrap the response text every 20 words
        wrapped_response = wrap_text_by_words(llm_call.response, words_per_line=20)
        print(wrapped_response)
    print("─" * 80)

def create_results_dataframe(llm_calls):
    """Create results DataFrame from LLM calls"""
    results_data = []
    for llm in llm_calls:
        results_data.append({
            'Model': llm.name,
            'Model_ID': llm.model_id,
            'Region': get_region_display(llm),
            'Status': llm.status,
            'Cost_Cents': round(llm.cost * 100, 4),
            'Latency_s': round(llm.latency, 2),
            'TTFT_s': round(llm.ttft, 2),
            'Input_Tokens': llm.input_tokens,
            'Output_Tokens': llm.output_tokens,
            'Total_Tokens': llm.input_tokens + llm.output_tokens,
            'Word_Count': llm.word_count,
            'Tokens_Per_Word': round(llm.tokens_per_word, 2),
            'Throughput_tokens_per_sec': round(llm.throughput_tokens_per_sec, 1),
            'Throughput_words_per_sec': round(llm.throughput_words_per_sec, 1),
            'Pricing_Type': 'Real' if llm.pricing_info else 'Estimated',
            'Input_Rate_Per_1M': llm.pricing_info['input'] if llm.pricing_info else 'N/A',
            'Output_Rate_Per_1M': llm.pricing_info['output'] if llm.pricing_info else 'N/A',
            'Response': llm.response
        })
    
    return pd.DataFrame(results_data)

def compare_models_simple(models, prompt, pricing_data=None, timeout_seconds=30):
    """
    Main function for comparing models in Jupyter notebooks
    
    Args:
        models: List of model IDs or name:model_id strings
        prompt: The question/prompt to send to models
        pricing_data: Optional bedrock_pricing_json dict for real pricing
        timeout_seconds: Timeout for each model call (default: 30 seconds)
    
    Returns:
        DataFrame with results for further analysis
    """
    
    # Parse model configurations
    model_configs = [parse_model_input(m) for m in models]
    
    print(f"🚀 Comparing {len(model_configs)} models...")
    print(f"📝 Question: {prompt}")
    print("=" * 80)
    
    # Display KPI explanations
    print("📊 KEY PERFORMANCE INDICATORS (KPIs):")
    print("🕐 LATENCY (s):      Total time from request to complete response")
    print("⚡ TTFT (s):         Time To First Token - responsiveness")
    print("📝 TOKENS:           Input/Output token counts")
    print("🔤 TOKENS/WORD:      Tokenization efficiency (lower is better)")
    print("🚀 THROUGHPUT:       Tokens and words generated per second (higher is better)")
    print("💰 COST (¢):         Actual cost based on AWS Bedrock pricing")
    print("🌍 REGION FALLBACK:  us-east-1 → us-west-2, plus us./eu./ap. prefixes")
    print("=" * 80)
    print()
    
    # Set pricing data if provided
    if pricing_data:
        set_pricing_data(pricing_data)
    else:
        print("💰 Using estimated pricing (no pricing data provided)")
    print()
    
    # Create LLM calls and run models sequentially
    llm_calls = [LLMCall(config["name"], config["model_id"]) for config in model_configs]
    
    print("⏳ Running models sequentially...")
    start_time = time.time()
    
    for i, llm_call in enumerate(llm_calls, 1):
        print(f"🔄 Running {i}/{len(llm_calls)}: {llm_call.name}")
        call_bedrock_sync(llm_call, prompt, pricing_data, timeout_seconds)
        display_model_result(llm_call, i, len(llm_calls))
    
    total_time = time.time() - start_time
    print(f"✅ All models completed in {total_time:.2f}s!")
    print()
    
    # Create and display results DataFrame
    df = create_results_dataframe(llm_calls)
    
    # print("📊 SUMMARY TABLE (ordered by increasing cost):")
    summary_df = df.drop('Response', axis=1).sort_values('Cost_Cents', ascending=True)
    # display(summary_df)
    
    # Performance highlights
    completed_df = df[df['Status'] == 'Complete']
    if not completed_df.empty:
        fastest_overall = completed_df.loc[completed_df['Latency_s'].idxmin()]
        fastest_ttft = completed_df.loc[completed_df['TTFT_s'].idxmin()]
        cheapest = completed_df.loc[completed_df['Cost_Cents'].idxmin()]
        most_efficient = completed_df.loc[completed_df['Tokens_Per_Word'].idxmin()]
        highest_throughput_tokens = completed_df.loc[completed_df['Throughput_tokens_per_sec'].idxmax()]
        highest_throughput_words = completed_df.loc[completed_df['Throughput_words_per_sec'].idxmax()]
        
        print("\n🏆 PERFORMANCE HIGHLIGHTS:")
        print(f"⚡ Fastest overall: {fastest_overall['Model']} ({fastest_overall['Latency_s']}s)")
        print(f"⚡ Fastest TTFT: {fastest_ttft['Model']} ({fastest_ttft['TTFT_s']}s)")
        print(f"🚀 Highest throughput tokens: {highest_throughput_tokens['Model']} ({highest_throughput_tokens['Throughput_tokens_per_sec']:.1f} tokens/sec)")
        print(f"🚀 Highest throughput words: {highest_throughput_words['Model']} ({highest_throughput_words['Throughput_words_per_sec']:.1f} words/sec)")
        print(f"💰 Cheapest: {cheapest['Model']} ({cheapest['Cost_Cents']:.4f}¢)")
        print(f"🔤 Most efficient: {most_efficient['Model']} ({most_efficient['Tokens_Per_Word']:.2f} tokens/word)")
    
    return df

def analyze_results(df):
    """Analyze comparison results with detailed insights"""
    
    print("📈 DETAILED ANALYSIS:")
    print("="*50)
    
    completed_df = df[df['Status'] == 'Complete']
    if completed_df.empty:
        print("❌ No successful completions to analyze")
        return
    
    print(f"✅ Successful completions: {len(completed_df)}/{len(df)}")
    print(f"📊 Average latency: {completed_df['Latency_s'].mean():.2f}s")
    print(f"📊 Average tokens per word: {completed_df['Tokens_Per_Word'].mean():.2f}")
    print(f"📊 Average throughput: {completed_df['Throughput_tokens_per_sec'].mean():.1f} tokens/sec, {completed_df['Throughput_words_per_sec'].mean():.1f} words/sec")
    print(f"📊 Average cost: {completed_df['Cost_Cents'].mean():.4f}¢")
    
    # Cost efficiency
    completed_df = completed_df.copy()
    completed_df['Output_Tokens_per_Dollar'] = completed_df['Output_Tokens'] / completed_df['Cost_Cents']
    
    print(f"\n💰 COST EFFICIENCY:")
    efficiency_df = completed_df[['Model', 'Output_Tokens_per_Dollar', 'Cost_Cents']].sort_values('Output_Tokens_per_Dollar', ascending=False)
    display(efficiency_df)
    
    print(f"\n⚡ SPEED ANALYSIS:")
    speed_df = completed_df[['Model', 'Latency_s', 'TTFT_s', 'Throughput_tokens_per_sec', 'Throughput_words_per_sec']].sort_values('Latency_s')
    display(speed_df)
    
    print(f"\n🔤 TOKENIZATION EFFICIENCY:")
    tokenization_df = completed_df[['Model', 'Word_Count', 'Output_Tokens', 'Tokens_Per_Word']].sort_values('Tokens_Per_Word')
    display(tokenization_df)
    
    return completed_df

# Set pandas display options for better Jupyter output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

if __name__ == "__main__":
    print("🚀 Minimal LLM Comparison Tool")
    print("Usage: from llm_compare_jupyter_clean import compare_models_simple")
    print("df = compare_models_simple(models, prompt, bedrock_pricing_json)")
