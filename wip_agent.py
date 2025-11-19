import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, computed_field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "" 

# Initialize the NEW 2025 Client for native PDF reading
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-2.0-flash-exp"  # or "gemini-1.5-pro"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0
)

# ==========================================
# 2. THE DATA MODELS - EXPANDED FOR FULL WIP
# ==========================================

class FullWipRow(BaseModel):
    """Complete WIP row with all standard columns"""
    job_id: str = Field(description="Job Number or ID")
    job_name: Optional[str] = Field(default="", description="Job Name or Description")
    
    # Contract and Cost columns
    total_contract_price: float = Field(description="Total Contract Price/Revenue")
    estimated_total_costs: float = Field(description="Estimated Total Costs at Completion")
    estimated_gross_profit: float = Field(description="Estimated Gross Profit")
    
    # Earned/Actual columns
    revenues_earned: float = Field(description="Revenues Earned to Date")
    cost_to_date: float = Field(description="Costs Incurred to Date")
    gross_profit_to_date: float = Field(description="Gross Profit to Date")
    
    # Billing columns
    billed_to_date: float = Field(description="Amount Billed to Date")
    cost_to_complete: float = Field(description="Cost to Complete")
    under_billings: float = Field(description="Under Billings")
    over_billings: float = Field(description="Over Billings")

class WipTotals(BaseModel):
    """Totals row from the WIP for validation"""
    total_contract_price: Optional[float] = None
    estimated_total_costs: Optional[float] = None
    estimated_gross_profit: Optional[float] = None
    revenues_earned: Optional[float] = None
    cost_to_date: Optional[float] = None
    gross_profit_to_date: Optional[float] = None
    billed_to_date: Optional[float] = None
    cost_to_complete: Optional[float] = None
    under_billings: Optional[float] = None
    over_billings: Optional[float] = None

class CalculatedWipRow(FullWipRow):
    """Enhanced with calculated fields for backwards compatibility"""
    
    @computed_field
    @property
    def contract_amount(self) -> float:
        """Legacy field for compatibility"""
        return self.total_contract_price
    
    @computed_field
    @property
    def est_cost(self) -> float:
        """Legacy field for compatibility"""
        return self.estimated_total_costs
    
    @computed_field
    @property
    def percent_complete(self) -> float:
        if self.estimated_total_costs and self.estimated_total_costs > 0:
            return round(self.cost_to_date / self.estimated_total_costs, 4)
        return 0.0
    
    @computed_field
    @property
    def earned_revenue(self) -> float:
        """Calculate if not provided"""
        if self.revenues_earned > 0:
            return self.revenues_earned
        return round(self.total_contract_price * self.percent_complete, 2)
    
    @computed_field
    @property
    def over_billing(self) -> float:
        """Use provided value or calculate"""
        if self.over_billings > 0:
            return self.over_billings
        val = self.billed_to_date - self.earned_revenue
        return round(val, 2) if val > 0 else 0.0
    
    @computed_field
    @property
    def under_billing(self) -> float:
        """Use provided value or calculate"""
        if self.under_billings > 0:
            return self.under_billings
        val = self.earned_revenue - self.billed_to_date
        return round(val, 2) if val > 0 else 0.0
    
    @computed_field
    @property
    def ctc(self) -> float:
        """Use provided value or calculate"""
        if self.cost_to_complete > 0:
            return self.cost_to_complete
        return round(self.estimated_total_costs - self.cost_to_date, 2)

class WipExtractionResult(BaseModel):
    """Complete extraction result with rows and totals"""
    rows: List[FullWipRow]
    totals: Optional[WipTotals] = None

class WipState(BaseModel):
    file_path: str
    processed_data: List[CalculatedWipRow] = []
    totals_row: Optional[WipTotals] = None
    final_json: Dict[str, Any] = {}

# ==========================================
# 3. THE EXTRACTION AGENT - FULL WIP
# ==========================================

def extractor_node(state: WipState):
    print(f"\n--- EXTRACTING COMPLETE WIP TABLE FROM: {state.file_path} ---")
    
    try:
        with open(state.file_path, "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found: {state.file_path}")
        return {"processed_data": [], "totals_row": None}

    # Comprehensive extraction prompt with multiple column mappings
    prompt = """
    Extract the COMPLETE WIP Schedule table including ALL columns and the TOTALS row.
    
    Return JSON matching this exact structure:
    {
        "rows": [
            {
                "job_id": "string",
                "job_name": "string",
                "total_contract_price": number,
                "estimated_total_costs": number,
                "estimated_gross_profit": number,
                "revenues_earned": number,
                "cost_to_date": number,
                "gross_profit_to_date": number,
                "billed_to_date": number,
                "cost_to_complete": number,
                "under_billings": number,
                "over_billings": number
            }
        ],
        "totals": {
            "total_contract_price": number,
            "estimated_total_costs": number,
            "estimated_gross_profit": number,
            "revenues_earned": number,
            "cost_to_date": number,
            "gross_profit_to_date": number,
            "billed_to_date": number,
            "cost_to_complete": number,
            "under_billings": number,
            "over_billings": number
        }
    }
    
    COLUMN MAPPINGS (use these to identify columns):
    
    For total_contract_price, look for:
    - "Total Contract", "Rev Contract", "Contract Amount", "Contract Price", 
    - "Revised Contract", "Contract Value", "Total Revenue"
    
    For estimated_total_costs, look for:
    - "Est Cost", "Estimated Cost", "Total Cost", "Est Total Cost",
    - "Estimated Costs", "Total Estimated Cost", "Budget Cost"
    
    For estimated_gross_profit, look for:
    - "Est GP", "Est Gross Profit", "Estimated GP", "Est Profit",
    - "Estimated Margin", "Contract Margin"
    
    For revenues_earned, look for:
    - "Earned Revenue", "Revenue Earned", "Earned", "Rev Earned",
    - "Revenue to Date", "Earned to Date"
    
    For cost_to_date, look for:
    - "Cost to Date", "JTD Cost", "Job to Date Cost", "Actual Cost",
    - "Cost Incurred", "Total Cost to Date", "CTD"
    
    For gross_profit_to_date, look for:
    - "GP to Date", "Gross Profit Earned", "Profit to Date", "GP Earned",
    - "Earned GP", "Actual GP"
    
    For billed_to_date, look for:
    - "Billed", "Total Billings", "Billed to Date", "Billings to Date",
    - "Invoice to Date", "Total Billed", "BTD"
    
    For cost_to_complete, look for:
    - "Cost to Complete", "CTC", "Remaining Cost", "Est to Complete",
    - "ETC", "Forecast to Complete"
    
    For under_billings, look for:
    - "Under Billing", "Underbilling", "UB", "Under Billed",
    - "Revenue in Excess", "Unbilled"
    
    For over_billings, look for:
    - "Over Billing", "Overbilling", "OB", "Over Billed",
    - "Billing in Excess", "Deferred Revenue"
    
    EXTRACTION RULES:
    1. Extract ALL data rows (jobs) - do NOT skip any
    2. SEPARATELY extract the TOTALS row (usually at bottom, labeled "Total", "Totals", "Grand Total", etc.)
    3. Convert parentheses (123.45) to negative: -123.45
    4. Convert percentage values to decimals (e.g., 15% → 0.15) if stored as percentages
    5. Remove currency symbols and commas from numbers
    6. If a column doesn't exist, use 0 for that field
    7. Include job names/descriptions if available
    
    Return ONLY valid JSON, no markdown or explanations.
    """

    try:
        # Generate content with structured output
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        
        # Parse the response
        extracted_data = json.loads(response.text)
        
        # Process rows
        raw_rows = [FullWipRow(**row) for row in extracted_data.get("rows", [])]
        calculated_rows = [CalculatedWipRow(**r.model_dump()) for r in raw_rows]
        
        # Process totals
        totals = None
        if "totals" in extracted_data and extracted_data["totals"]:
            totals = WipTotals(**extracted_data["totals"])
        
        print(f"--- EXTRACTED {len(calculated_rows)} ROWS ---")
        if totals:
            print(f"--- TOTALS ROW EXTRACTED FOR VALIDATION ---")
        
        return {
            "processed_data": calculated_rows,
            "totals_row": totals
        }
        
    except Exception as e:
        print(f"EXTRACTION ERROR: {e}")
        return {"processed_data": [], "totals_row": None}

# ==========================================
# 4. THE ANALYSIS AGENT - KEEP ORIGINAL STRUCTURE
# ==========================================

def analyst_node(state: WipState):
    print("--- ANALYZING COMPLETE WIP DATA ---")
    data = state.processed_data
    totals = state.totals_row
    
    if not data:
        print("NO DATA. SKIPPING ANALYSIS.")
        return {"final_json": {}}

    # Calculate aggregations
    t_contract = sum(r.total_contract_price for r in data)
    t_cost = sum(r.estimated_total_costs for r in data)
    t_earned = sum(r.revenues_earned for r in data if r.revenues_earned > 0) or sum(r.earned_revenue for r in data)
    t_cost_to_date = sum(r.cost_to_date for r in data)
    t_billed = sum(r.billed_to_date for r in data)
    t_ctc = sum(r.cost_to_complete for r in data if r.cost_to_complete > 0) or sum(r.ctc for r in data)
    t_under = sum(r.under_billings for r in data)
    t_over = sum(r.over_billings for r in data)
    
    # Calculate metrics - keep original logic
    t_uegp = sum((r.contract_amount - r.est_cost) - (r.earned_revenue - r.cost_to_date) for r in data)
    gp_pct = ((t_earned - t_cost_to_date) / t_earned * 100) if t_earned else 0
    
    # Net billing position
    net_billing = t_over - t_under
    net_billing_label = f"Over ${abs(net_billing)/1000:.0f}K" if net_billing > 0 else f"Under ${abs(net_billing)/1000:.0f}K"
    
    # Risk analysis - keep original structure
    risks = []
    sorted_jobs = sorted(data, key=lambda x: max(x.over_billing, x.under_billing), reverse=True)
    
    for job in sorted_jobs[:5]:  # Keep top 5 as original
        variance = max(job.over_billing, job.under_billing)
        severity = "high" if variance > 100000 else "medium"
        risks.append({
            "id": job.job_id,
            "jobId": job.job_id,
            "jobName": job.job_name or "",  # Add name if available
            "riskTags": "Overbilling" if job.over_billing > 0 else "Underbilling",
            "riskLevel": severity,
            "riskLevelLabel": severity.capitalize(),
            "amountAbs": f"${variance:,.0f}",
            "ubobType": "OB" if job.over_billing > 0 else "UB",
            "percentComplete": f"{job.percent_complete:.1%}"
        })

    # Generate narrative - keep it simple
    response = llm.invoke(f"Write 1 concise, professional sentence summarizing the portfolio health. Total Contract: ${t_contract:,.0f}. GP: {gp_pct:.1f}%. Risks: {len(risks)} jobs.")
    narrative = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    narrative = narrative.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Build the payload - KEEP ORIGINAL STRUCTURE
    payload = {
        # Full extracted table with all columns (new)
        "full_wip_table": [row.model_dump() for row in data],
        
        # Keep original clean_table structure exactly
        "clean_table": [row.model_dump() for row in data],
        
        # Validation data if we have totals
        "totals_validation": totals.model_dump() if totals else None,
        
        # Widget data - KEEP ORIGINAL STRUCTURE
        "widget_data": {
            # Summary stays inside widget_data as original
            "summary": {"type": "text", "text": narrative},
            
            # Metrics - restore original order and labels
            "metrics": {
                "totalContractAmount": {
                    "label": "Total Contract", 
                    "value": f"${t_contract/1000000:.2f}M"
                },
                "uegp": {
                    "label": "UEGP", 
                    "value": f"${t_uegp/1000000:.2f}M"
                },
                "gpPct": {
                    "label": "GP%", 
                    "value": f"{gp_pct:.1f}%"
                },
                "ctc": {
                    "label": "CTC", 
                    "value": f"${t_ctc/1000000:.2f}M"
                },
                "wipGpPct": {
                    "label": "WIP GP%", 
                    "value": f"{gp_pct:.1f}%"
                },
                # Replace CC GP% with Net Billing Position as you suggested
                "ccGpPct": {
                    "label": "Net Billing", 
                    "value": net_billing_label
                }
            },
            "riskRowsAll": risks
        }
    }
    
    return {"final_json": payload}

# ==========================================
# 5. EXECUTION FLOW
# ==========================================

workflow = StateGraph(WipState)
workflow.add_node("extract", extractor_node)
workflow.add_node("analyze", analyst_node)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "analyze")
workflow.add_edge("analyze", END)
app = workflow.compile()

if __name__ == "__main__":
    TEST_FILE = "test_wip.pdf" 
    if os.path.exists(TEST_FILE):
        result = app.invoke({"file_path": TEST_FILE})
        
        if result.get("final_json"):
            print("\n--- EXTRACTION COMPLETE ---")
            print(f"Rows extracted: {len(result['final_json'].get('full_wip_table', []))}")
            
            print("\n--- WIDGET DATA (for frontend) ---")
            print(json.dumps(result["final_json"]["widget_data"], indent=2))
            
            # Save full output for debugging
            with open("wip_output.json", "w") as f:
                json.dump(result["final_json"], f, indent=2)
            print("\nFull output saved to wip_output.json")
    else:
        print(f"ERROR: Please put a file named '{TEST_FILE}' in this folder.")