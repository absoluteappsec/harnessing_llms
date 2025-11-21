"""
Advanced Agentic Evaluation System: Battle of the Bots with LLM Judge

This system implements a sophisticated evaluation framework where:
1. Multiple agents compete to analyze code security
2. An LLM judge evaluates their performance
3. Comprehensive scoring and metrics are provided
4. Ground truth validation ensures accuracy

Architecture:
- Agent 1: First LLM model being tested for vulnerability identification
- Agent 2: Second LLM model being tested for vulnerability identification
- Judge LLM: Evaluates both models' responses for accuracy, reasoning quality, and completeness
"""

import json
import asyncio
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from datetime import datetime

from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate

# Removed JsonOutputParser - using simple text parsing instead
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""

    name: str
    model_id: str
    temperature: float = 0.5
    description: str = ""


class AgentType(Enum):
    AGENT_1 = "agent_1"
    AGENT_2 = "agent_2"
    JUDGE = "judge"


@dataclass
class TestCase:
    """Represents a code security test case with ground truth"""

    code: str
    description: str
    is_vulnerable: bool
    vulnerability_type: str
    severity: str  # low, medium, high, critical
    cwe_id: str = None
    explanation: str = ""


@dataclass
class AgentResponse:
    """Represents an agent's analysis response"""

    agent_name: str
    model_id: str
    is_vulnerable: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    vulnerability_details: str
    suggested_fix: str = ""
    execution_time: float = 0.0


@dataclass
class JudgeEvaluation:
    """Represents the judge's evaluation of agent responses"""

    accuracy_score: float  # 0.0 to 1.0
    reasoning_quality: float  # 0.0 to 1.0
    completeness_score: float  # 0.0 to 1.0
    overall_score: float  # 0.0 to 1.0
    feedback: str
    correct_prediction: bool


@dataclass
class BattleResult:
    """Results of a battle between agents"""

    test_case: TestCase
    agent1_response: AgentResponse
    agent2_response: AgentResponse
    agent1_evaluation: JudgeEvaluation
    agent2_evaluation: JudgeEvaluation
    winner: str  # agent name
    battle_summary: str


class SecurityAgent:
    """LLM agent for security analysis"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = ChatBedrock(
            model_id=model_config.model_id,
            model_kwargs={"temperature": model_config.temperature},
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for security analysis"""
        return """You are an expert security analyst. Your role is to:
        - Thoroughly analyze code for potential security vulnerabilities
        - Provide detailed reasoning for your conclusions
        - Identify specific vulnerability types and their severity
        - Suggest practical remediation steps
        - Be accurate in your vulnerability assessment
        
        Analyze the code carefully and provide a comprehensive security assessment."""

    async def analyze_code(self, test_case: TestCase) -> AgentResponse:
        """Analyze code and return structured response"""
        import time

        start_time = time.time()

        prompt = PromptTemplate.from_template(
            """
        {system_prompt}
        
        ## Code to Analyze:
        ```python
        {code}
        ```
        
        ## Context:
        {description}
        
        ## Instructions:
        Analyze this code for security vulnerabilities. Provide your response in this exact format:
        
        VULNERABLE: [Yes/No]
        CONFIDENCE: [0.0-1.0]
        REASONING: [Your detailed analysis]
        VULNERABILITIES: [Specific issues found, or "None" if secure]
        SUGGESTED_FIX: [Remediation steps, or "None needed" if secure]
        
        Be thorough in your analysis and reasoning.
        """
        )

        formatted_prompt = prompt.format(
            system_prompt=self.get_system_prompt(),
            code=test_case.code,
            description=test_case.description,
        )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])
            response_text = response.content

            execution_time = time.time() - start_time

            # Simple text parsing
            is_vulnerable = self._extract_vulnerable(response_text)
            confidence = self._extract_confidence(response_text)
            reasoning = self._extract_field(response_text, "REASONING")
            vulnerability_details = self._extract_field(
                response_text, "VULNERABILITIES"
            )
            suggested_fix = self._extract_field(response_text, "SUGGESTED_FIX")

            return AgentResponse(
                agent_name=self.model_config.name,
                model_id=self.model_config.model_id,
                is_vulnerable=is_vulnerable,
                confidence=confidence,
                reasoning=reasoning,
                vulnerability_details=vulnerability_details,
                suggested_fix=suggested_fix,
                execution_time=execution_time,
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.model_config.name,
                model_id=self.model_config.model_id,
                is_vulnerable=False,
                confidence=0.0,
                reasoning=f"Error in analysis: {str(e)}",
                vulnerability_details="Analysis failed",
                suggested_fix="",
                execution_time=time.time() - start_time,
            )

    def _extract_vulnerable(self, text: str) -> bool:
        """Extract vulnerability assessment from response"""
        text_upper = text.upper()
        if "VULNERABLE: YES" in text_upper or "VULNERABLE:YES" in text_upper:
            return True
        elif "VULNERABLE: NO" in text_upper or "VULNERABLE:NO" in text_upper:
            return False
        # Fallback: look for keywords
        elif any(
            word in text_upper for word in ["VULNERABLE", "INSECURE", "EXPLOITABLE"]
        ):
            return True
        return False

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from response"""
        import re

        # Look for CONFIDENCE: X.X pattern
        match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if match:
            try:
                conf = float(match.group(1))
                return min(max(conf, 0.0), 1.0)  # Clamp between 0 and 1
            except ValueError:
                pass
        # Fallback: look for confidence keywords
        text_upper = text.upper()
        if any(
            word in text_upper
            for word in ["HIGH CONFIDENCE", "VERY CONFIDENT", "CERTAIN"]
        ):
            return 0.9
        elif any(
            word in text_upper for word in ["MEDIUM CONFIDENCE", "MODERATELY CONFIDENT"]
        ):
            return 0.7
        elif any(word in text_upper for word in ["LOW CONFIDENCE", "UNCERTAIN"]):
            return 0.4
        return 0.5  # Default moderate confidence

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a specific field from the response"""
        import re

        pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Not provided"


class LLMJudge:
    """LLM Judge that evaluates agent responses"""

    def __init__(self, model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
        # Use a more powerful model for judging
        self.llm = ChatBedrock(
            model_id=model_id,
            model_kwargs={
                "temperature": 0.1
            },  # Low temperature for consistent evaluation
        )

    async def evaluate_response(
        self, test_case: TestCase, agent_response: AgentResponse
    ) -> JudgeEvaluation:
        """Evaluate an agent's response against ground truth"""

        prompt = PromptTemplate.from_template(
            """
        You are an expert security evaluation judge. Evaluate the following agent's security analysis.
        
        ## Ground Truth:
        - Code is vulnerable: {is_vulnerable}
        - Vulnerability type: {vulnerability_type}
        - Severity: {severity}
        - CWE ID: {cwe_id}
        - Explanation: {explanation}
        
        ## Agent Response to Evaluate:
        - Model: {model_name} ({model_id})
        - Predicted vulnerable: {predicted_vulnerable}
        - Confidence: {confidence}
        - Reasoning: {reasoning}
        - Vulnerability details: {vulnerability_details}
        - Suggested fix: {suggested_fix}
        
        ## Code Analyzed:
        ```python
        {code}
        ```
        
        ## Evaluation Criteria:
        1. **Accuracy (0.0-1.0)**: How well does the prediction match ground truth?
        2. **Reasoning Quality (0.0-1.0)**: How sound and detailed is the reasoning?
        3. **Completeness (0.0-1.0)**: How thorough is the analysis?
        
        ## Instructions:
        Provide your evaluation in this exact format:
        
        ACCURACY_SCORE: [0.0-1.0]
        REASONING_QUALITY: [0.0-1.0]
        COMPLETENESS_SCORE: [0.0-1.0]
        OVERALL_SCORE: [0.0-1.0]
        CORRECT_PREDICTION: [Yes/No]
        FEEDBACK: [Detailed explanation of scores]
        
        Evaluate the model's performance objectively based on accuracy and reasoning quality.
        Consider both precision (avoiding false positives) and recall (catching real vulnerabilities).
        """
        )

        formatted_prompt = prompt.format(
            is_vulnerable=test_case.is_vulnerable,
            vulnerability_type=test_case.vulnerability_type,
            severity=test_case.severity,
            cwe_id=test_case.cwe_id or "N/A",
            explanation=test_case.explanation,
            model_name=agent_response.agent_name,
            model_id=agent_response.model_id,
            predicted_vulnerable=agent_response.is_vulnerable,
            confidence=agent_response.confidence,
            reasoning=agent_response.reasoning,
            vulnerability_details=agent_response.vulnerability_details,
            suggested_fix=agent_response.suggested_fix,
            code=test_case.code,
        )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])
            response_text = response.content

            # Simple text parsing for judge evaluation
            accuracy_score = self._extract_score(response_text, "ACCURACY_SCORE")
            reasoning_quality = self._extract_score(response_text, "REASONING_QUALITY")
            completeness_score = self._extract_score(
                response_text, "COMPLETENESS_SCORE"
            )
            overall_score = self._extract_score(response_text, "OVERALL_SCORE")
            correct_prediction = self._extract_correct_prediction(response_text)
            feedback = self._extract_field(response_text, "FEEDBACK")

            return JudgeEvaluation(
                accuracy_score=accuracy_score,
                reasoning_quality=reasoning_quality,
                completeness_score=completeness_score,
                overall_score=overall_score,
                feedback=feedback,
                correct_prediction=correct_prediction,
            )
        except Exception as e:
            return JudgeEvaluation(
                accuracy_score=0.0,
                reasoning_quality=0.0,
                completeness_score=0.0,
                overall_score=0.0,
                feedback=f"Evaluation error: {str(e)}",
                correct_prediction=False,
            )

    def _extract_score(self, text: str, field_name: str) -> float:
        """Extract a numeric score from the response"""
        import re

        pattern = rf"{field_name}:\s*([0-9]*\.?[0-9]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            except ValueError:
                pass
        return 0.0

    def _extract_correct_prediction(self, text: str) -> bool:
        """Extract correct prediction boolean from response"""
        text_upper = text.upper()
        if (
            "CORRECT_PREDICTION: YES" in text_upper
            or "CORRECT_PREDICTION:YES" in text_upper
        ):
            return True
        elif (
            "CORRECT_PREDICTION: NO" in text_upper
            or "CORRECT_PREDICTION:NO" in text_upper
        ):
            return False
        return False

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a specific field from the response"""
        import re

        pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Not provided"


class AgenticBattleSystem:
    """Main system orchestrating the LLM model comparison"""

    def __init__(
        self,
        model1_config: ModelConfig,
        model2_config: ModelConfig,
        judge_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    ):
        self.agent1 = SecurityAgent(model1_config)
        self.agent2 = SecurityAgent(model2_config)
        self.judge = LLMJudge(judge_model_id)
        self.battle_results: List[BattleResult] = []
        self.model1_config = model1_config
        self.model2_config = model2_config

    def get_test_cases(self) -> List[TestCase]:
        """Get comprehensive test cases for evaluation"""
        return [
            TestCase(
                code="""
@login_required
@user_passes_test(can_create_project)
def update_user_active(request):
    user_id = request.GET.get('user_id')
    User.objects.filter(id=user_id).update(is_active=False)
                """,
                description="Django view that updates user active status",
                is_vulnerable=True,
                vulnerability_type="Insecure Direct Object Reference (IDOR)",
                severity="high",
                cwe_id="CWE-639",
                explanation="User can modify any user's active status by changing user_id parameter",
            ),
            TestCase(
                code="""
def get_user_profile(request):
    user = request.user
    if user.is_authenticated:
        profile = UserProfile.objects.get(user=user)
        return JsonResponse({'profile': profile.to_dict()})
    return JsonResponse({'error': 'Not authenticated'}, status=401)
                """,
                description="Function to get authenticated user's profile",
                is_vulnerable=False,
                vulnerability_type="None",
                severity="low",
                explanation="Properly checks authentication and only returns current user's profile",
            ),
            TestCase(
                code="""
def transfer_funds(request):
    amount = float(request.POST.get('amount'))
    to_account = request.POST.get('to_account')
    from_account = request.user.account
    
    if from_account.balance >= amount:
        from_account.balance -= amount
        to_account_obj = Account.objects.get(number=to_account)
        to_account_obj.balance += amount
        from_account.save()
        to_account_obj.save()
        return JsonResponse({'success': True})
    return JsonResponse({'error': 'Insufficient funds'})
                """,
                description="Banking function to transfer funds between accounts",
                is_vulnerable=True,
                vulnerability_type="Race Condition / TOCTOU",
                severity="critical",
                cwe_id="CWE-367",
                explanation="No transaction isolation, vulnerable to race conditions in concurrent transfers",
            ),
            TestCase(
                code="""
import subprocess

def ping_host(request):
    host = request.GET.get('host', 'localhost')
    if re.match(r'^[a-zA-Z0-9.-]+$', host):
        result = subprocess.run(['ping', '-c', '1', host], 
                              capture_output=True, text=True)
        return JsonResponse({'output': result.stdout})
    return JsonResponse({'error': 'Invalid host format'})
                """,
                description="Network utility to ping hosts with input validation",
                is_vulnerable=False,
                vulnerability_type="None",
                severity="low",
                explanation="Proper input validation prevents command injection",
            ),
            TestCase(
                code="""
def execute_query(request):
    query = request.POST.get('sql_query')
    if 'admin' in request.user.groups.all().values_list('name', flat=True):
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return JsonResponse({'results': results})
    return JsonResponse({'error': 'Admin access required'})
                """,
                description="Admin function to execute custom SQL queries",
                is_vulnerable=True,
                vulnerability_type="SQL Injection",
                severity="critical",
                cwe_id="CWE-89",
                explanation="Direct SQL execution without parameterization allows SQL injection",
            ),
        ]

    async def run_battle(self, test_case: TestCase) -> BattleResult:
        """Run a single battle between defender and attacker agents"""
        print(f"\n🥊 Battle: {test_case.description}")
        print(f"Ground Truth: {'VULNERABLE' if test_case.is_vulnerable else 'SECURE'}")

        # Get responses from both agents
        agent1_response, agent2_response = await asyncio.gather(
            self.agent1.analyze_code(test_case), self.agent2.analyze_code(test_case)
        )

        print(
            f"🤖 {self.model1_config.name}: {'VULNERABLE' if agent1_response.is_vulnerable else 'SECURE'} (confidence: {agent1_response.confidence:.2f})"
        )
        print(
            f"🤖 {self.model2_config.name}: {'VULNERABLE' if agent2_response.is_vulnerable else 'SECURE'} (confidence: {agent2_response.confidence:.2f})"
        )

        # Get judge evaluations
        agent1_eval, agent2_eval = await asyncio.gather(
            self.judge.evaluate_response(test_case, agent1_response),
            self.judge.evaluate_response(test_case, agent2_response),
        )

        # Determine winner
        winner = (
            self.model1_config.name
            if agent1_eval.overall_score > agent2_eval.overall_score
            else self.model2_config.name
        )

        battle_summary = f"""
        Winner: {winner}
        {self.model1_config.name} Score: {agent1_eval.overall_score:.3f} (Correct: {agent1_eval.correct_prediction})
        {self.model2_config.name} Score: {agent2_eval.overall_score:.3f} (Correct: {agent2_eval.correct_prediction})
        """

        print(f"🏆 {battle_summary}")

        return BattleResult(
            test_case=test_case,
            agent1_response=agent1_response,
            agent2_response=agent2_response,
            agent1_evaluation=agent1_eval,
            agent2_evaluation=agent2_eval,
            winner=winner,
            battle_summary=battle_summary,
        )

    async def run_tournament(self) -> Dict[str, Any]:
        """Run complete tournament and generate comprehensive results"""
        print("🚀 Starting Agentic Security Battle Tournament!")
        print("=" * 60)

        test_cases = self.get_test_cases()

        # Run all battles
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[Battle {i}/{len(test_cases)}]")
            battle_result = await self.run_battle(test_case)
            self.battle_results.append(battle_result)

        # Calculate tournament statistics
        return self.calculate_tournament_stats()

    def calculate_tournament_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive tournament statistics"""
        agent1_scores = [
            br.agent1_evaluation.overall_score for br in self.battle_results
        ]
        agent2_scores = [
            br.agent2_evaluation.overall_score for br in self.battle_results
        ]

        agent1_wins = sum(
            1 for br in self.battle_results if br.winner == self.model1_config.name
        )
        agent2_wins = sum(
            1 for br in self.battle_results if br.winner == self.model2_config.name
        )

        agent1_accuracy = sum(
            1 for br in self.battle_results if br.agent1_evaluation.correct_prediction
        )
        agent2_accuracy = sum(
            1 for br in self.battle_results if br.agent2_evaluation.correct_prediction
        )

        stats = {
            "tournament_summary": {
                "total_battles": len(self.battle_results),
                "model1_name": self.model1_config.name,
                "model2_name": self.model2_config.name,
                "model1_wins": agent1_wins,
                "model2_wins": agent2_wins,
                "win_rate_model1": agent1_wins / len(self.battle_results),
                "win_rate_model2": agent2_wins / len(self.battle_results),
            },
            "performance_metrics": {
                self.model1_config.name: {
                    "model_id": self.model1_config.model_id,
                    "avg_score": statistics.mean(agent1_scores),
                    "accuracy": agent1_accuracy / len(self.battle_results),
                    "score_std": (
                        statistics.stdev(agent1_scores) if len(agent1_scores) > 1 else 0
                    ),
                },
                self.model2_config.name: {
                    "model_id": self.model2_config.model_id,
                    "avg_score": statistics.mean(agent2_scores),
                    "accuracy": agent2_accuracy / len(self.battle_results),
                    "score_std": (
                        statistics.stdev(agent2_scores) if len(agent2_scores) > 1 else 0
                    ),
                },
            },
            "detailed_results": [asdict(br) for br in self.battle_results],
            "timestamp": datetime.now().isoformat(),
        }

        return stats

    def print_tournament_summary(self, stats: Dict[str, Any]):
        """Print formatted tournament summary"""
        print("\n" + "=" * 60)
        print("🏆 LLM MODEL COMPARISON RESULTS")
        print("=" * 60)

        summary = stats["tournament_summary"]
        model1_metrics = stats["performance_metrics"][summary["model1_name"]]
        model2_metrics = stats["performance_metrics"][summary["model2_name"]]

        print(f"\n📊 Overall Results:")
        print(f"   Total Battles: {summary['total_battles']}")
        print(
            f"   {summary['model1_name']} Wins: {summary['model1_wins']} ({summary['win_rate_model1']:.1%})"
        )
        print(
            f"   {summary['model2_name']} Wins: {summary['model2_wins']} ({summary['win_rate_model2']:.1%})"
        )

        print(f"\n🤖 {summary['model1_name']} Performance:")
        print(f"   Model ID: {model1_metrics['model_id']}")
        print(f"   Average Score: {model1_metrics['avg_score']:.3f}")
        print(f"   Accuracy: {model1_metrics['accuracy']:.1%}")
        print(f"   Score Std Dev: {model1_metrics['score_std']:.3f}")

        print(f"\n🤖 {summary['model2_name']} Performance:")
        print(f"   Model ID: {model2_metrics['model_id']}")
        print(f"   Average Score: {model2_metrics['avg_score']:.3f}")
        print(f"   Accuracy: {model2_metrics['accuracy']:.1%}")
        print(f"   Score Std Dev: {model2_metrics['score_std']:.3f}")

        # Determine overall champion
        if summary["win_rate_model1"] > summary["win_rate_model2"]:
            champion = f"🏆 {summary['model1_name']}"
        elif summary["win_rate_model2"] > summary["win_rate_model1"]:
            champion = f"🏆 {summary['model2_name']}"
        else:
            champion = "🤝 TIE"

        print(f"\n🎉 TOURNAMENT CHAMPION: {champion}")
        print("=" * 60)


# Predefined model configurations for easy comparison
AVAILABLE_MODELS = {
    "claude_haiku": ModelConfig(
        name="Claude-3.5-Haiku",
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        temperature=0.3,
        description="Fast, efficient model good for structured tasks",
    ),
    "claude_sonnet": ModelConfig(
        name="Claude-3.5-Sonnet",
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.3,
        description="Balanced model with strong reasoning capabilities",
    ),
    "claude_sonnet_v1": ModelConfig(
        name="Claude-3-Sonnet-v1",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.3,
        description="Previous version of Claude Sonnet for comparison",
    ),
    "titan_express": ModelConfig(
        name="Titan-Text-Express",
        model_id="amazon.titan-text-express-v1",
        temperature=0.3,
        description="Amazon's fast text generation model",
    ),
}


async def run_model_comparison(model1_key: str, model2_key: str) -> Dict[str, Any]:
    """Run comparison between two specific models"""
    if model1_key not in AVAILABLE_MODELS or model2_key not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model keys. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    model1_config = AVAILABLE_MODELS[model1_key]
    model2_config = AVAILABLE_MODELS[model2_key]

    print(f"\n🤖 Model Comparison: {model1_config.name} vs {model2_config.name}")
    print(f"Model 1: {model1_config.description}")
    print(f"Model 2: {model2_config.description}")

    battle_system = AgenticBattleSystem(model1_config, model2_config)

    try:
        # Run the tournament
        stats = await battle_system.run_tournament()

        # Print results
        battle_system.print_tournament_summary(stats)

        # Print detailed battle results to screen instead of saving to file
        print("\n" + "=" * 60)
        print("📋 DETAILED BATTLE RESULTS")
        print("=" * 60)

        for i, battle in enumerate(battle_system.battle_results, 1):
            print(f"\n[Battle {i}] {battle.test_case.description}")
            print(
                f"Ground Truth: {'VULNERABLE' if battle.test_case.is_vulnerable else 'SECURE'}"
            )
            print(f"Winner: {battle.winner}")
            print(f"\n{model1_config.name}:")
            print(
                f"  Prediction: {'VULNERABLE' if battle.agent1_response.is_vulnerable else 'SECURE'}"
            )
            print(f"  Confidence: {battle.agent1_response.confidence:.2f}")
            print(f"  Score: {battle.agent1_evaluation.overall_score:.3f}")
            print(f"  Correct: {battle.agent1_evaluation.correct_prediction}")
            print(f"\n{model2_config.name}:")
            print(
                f"  Prediction: {'VULNERABLE' if battle.agent2_response.is_vulnerable else 'SECURE'}"
            )
            print(f"  Confidence: {battle.agent2_response.confidence:.2f}")
            print(f"  Score: {battle.agent2_evaluation.overall_score:.3f}")
            print(f"  Correct: {battle.agent2_evaluation.correct_prediction}")
            print("-" * 40)

        return stats

    except Exception as e:
        print(f"❌ Tournament failed: {str(e)}")
        raise


async def main():
    """Main execution function with model selection"""
    print("🤖 LLM Security Analysis Comparison")
    print("=" * 50)
    print("\nAvailable models:")
    for key, config in AVAILABLE_MODELS.items():
        print(f"  {key}: {config.name} - {config.description}")

    # Default comparison: Haiku vs Sonnet
    print("\n🚀 Running default comparison: Claude Haiku vs Claude Sonnet")
    stats = await run_model_comparison("claude_haiku", "claude_sonnet")

    print("\n✅ Model comparison completed!")
    return stats


if __name__ == "__main__":
    # Run the model comparison
    results = asyncio.run(main())

    # Example of how to run custom comparisons:
    # results = asyncio.run(run_model_comparison("claude_haiku", "claude_sonnet"))
    # results = asyncio.run(run_model_comparison("claude_sonnet", "claude_opus"))
    # results = asyncio.run(run_model_comparison("claude_haiku", "titan_express"))
