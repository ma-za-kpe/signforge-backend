#!/usr/bin/env python3
"""
COMPREHENSIVE BRAIN TESTS
Tests all aspects of the GHSL brain including edge cases

Tests cover:
1. Single word searches
2. Multi-word phrase normalization
3. Typo tolerance
4. Case insensitivity
5. Autocomplete functionality
6. Edge cases and error handling
"""
import requests
import json
from typing import List, Dict, Tuple
from colorama import init, Fore, Style

init(autoreset=True)

BASE_URL = "http://localhost:9000"

class BrainTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = []

    def test(self, name: str, query: str, expected_word: str, min_confidence: float = 0.8):
        """Test a search query"""
        try:
            response = requests.get(f"{BASE_URL}/api/search", params={"q": query})

            if response.status_code != 200:
                self.fail(name, f"HTTP {response.status_code}")
                return

            data = response.json()
            matched_word = data['metadata']['matched_word']
            confidence = data['confidence']

            if matched_word == expected_word and confidence >= min_confidence:
                self.success(name, f"{matched_word} ({confidence:.2f})")
            else:
                self.fail(name, f"Got {matched_word} ({confidence:.2f}), expected {expected_word}")

        except Exception as e:
            self.fail(name, str(e))

    def test_autocomplete(self, name: str, query: str, should_contain: List[str]):
        """Test autocomplete suggestions"""
        try:
            response = requests.get(f"{BASE_URL}/api/autocomplete", params={"q": query})

            if response.status_code != 200:
                self.fail(name, f"HTTP {response.status_code}")
                return

            data = response.json()
            suggestions = data.get('suggestions', [])

            # Clean suggestions (remove hints)
            clean_suggestions = [s.split(' (')[0] for s in suggestions]

            missing = [word for word in should_contain if word not in clean_suggestions]

            if not missing:
                self.success(name, f"Found {should_contain}")
            else:
                self.fail(name, f"Missing {missing}, got {clean_suggestions[:5]}")

        except Exception as e:
            self.fail(name, str(e))

    def test_404(self, name: str, query: str):
        """Test that non-existent words return 404"""
        try:
            response = requests.get(f"{BASE_URL}/api/search", params={"q": query})

            if response.status_code == 404:
                self.success(name, "404 as expected")
            else:
                self.fail(name, f"Got HTTP {response.status_code}, expected 404")

        except Exception as e:
            self.fail(name, str(e))

    def success(self, name: str, details: str):
        """Record successful test"""
        self.passed += 1
        print(f"{Fore.GREEN}✅ {name}{Style.RESET_ALL}")
        print(f"   {details}\n")
        self.test_results.append({"name": name, "status": "PASS", "details": details})

    def fail(self, name: str, reason: str):
        """Record failed test"""
        self.failed += 1
        print(f"{Fore.RED}❌ {name}{Style.RESET_ALL}")
        print(f"   {reason}\n")
        self.test_results.append({"name": name, "status": "FAIL", "details": reason})

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0

        print("="*70)
        print(f"{Fore.CYAN}📊 TEST SUMMARY{Style.RESET_ALL}")
        print("="*70)
        print(f"Total Tests: {total}")
        print(f"{Fore.GREEN}Passed: {self.passed}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {self.failed}{Style.RESET_ALL}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("="*70)

        # Target is 99% accuracy
        if success_rate >= 99:
            print(f"{Fore.GREEN}🎉 TARGET ACHIEVED: 99%+ accuracy!{Style.RESET_ALL}")
        elif success_rate >= 95:
            print(f"{Fore.YELLOW}⚠️  GOOD: {success_rate:.1f}% (Target: 99%){Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ NEEDS IMPROVEMENT: {success_rate:.1f}% (Target: 99%){Style.RESET_ALL}")

def main():
    tester = BrainTester()

    print(f"{Fore.CYAN}🧪 COMPREHENSIVE BRAIN TESTS{Style.RESET_ALL}")
    print("="*70 + "\n")

    # ========== BASIC SINGLE WORD TESTS ==========
    print(f"{Fore.YELLOW}📝 BASIC SINGLE WORD TESTS{Style.RESET_ALL}\n")

    tester.test("Single: 'father'", "father", "FATHER", 0.9)
    tester.test("Single: 'mother'", "mother", "MOTHER", 0.9)
    tester.test("Single: 'school'", "school", "SCHOOL", 0.9)
    tester.test("Single: 'hello'", "hello", "HELLO", 0.9)
    tester.test("Single: 'thank'", "thank", "THANK", 0.9)

    # ========== MULTI-WORD PHRASE TESTS ==========
    print(f"{Fore.YELLOW}💬 MULTI-WORD PHRASE NORMALIZATION{Style.RESET_ALL}\n")

    tester.test("Phrase: 'thank you'", "thank you", "THANK", 0.99)
    tester.test("Phrase: 'good morning'", "good morning", "MORNING", 0.99)
    tester.test("Phrase: 'good afternoon'", "good afternoon", "AFTERNOON", 0.99)
    tester.test("Phrase: 'good evening'", "good evening", "EVENING", 0.99)
    tester.test("Phrase: 'my father'", "my father", "FATHER", 0.99)
    tester.test("Phrase: 'my mother'", "my mother", "MOTHER", 0.99)
    tester.test("Phrase: 'i love you'", "i love you", "LOVE", 0.99)
    tester.test("Phrase: 'you're welcome'", "you're welcome", "WELCOME", 0.99)
    tester.test("Phrase: 'excuse me'", "excuse me", "EXCUSE", 0.99)

    # ========== CASE INSENSITIVITY TESTS ==========
    print(f"{Fore.YELLOW}🔤 CASE INSENSITIVITY{Style.RESET_ALL}\n")

    tester.test("Upper: 'FATHER'", "FATHER", "FATHER", 0.9)
    tester.test("Lower: 'father'", "father", "FATHER", 0.9)
    tester.test("Mixed: 'FaThEr'", "FaThEr", "FATHER", 0.9)
    tester.test("Upper phrase: 'THANK YOU'", "THANK YOU", "THANK", 0.99)

    # ========== WHITESPACE HANDLING ==========
    print(f"{Fore.YELLOW}⎵  WHITESPACE HANDLING{Style.RESET_ALL}\n")

    tester.test("Leading space: ' father'", " father", "FATHER", 0.9)
    tester.test("Trailing space: 'father '", "father ", "FATHER", 0.9)
    tester.test("Extra spaces: 'thank  you'", "thank  you", "THANK", 0.99)

    # ========== TYPO TOLERANCE ==========
    print(f"{Fore.YELLOW}✏️  TYPO TOLERANCE{Style.RESET_ALL}\n")

    tester.test("Typo: 'scool' → 'SCHOOL'", "scool", "SCHOOL", 0.7)
    tester.test("Typo: 'fater' → 'FATHER'", "fater", "FATHER", 0.7)
    tester.test("Typo: 'moter' → 'MOTHER'", "moter", "MOTHER", 0.7)

    # ========== AUTOCOMPLETE TESTS ==========
    print(f"{Fore.YELLOW}🔍 AUTOCOMPLETE FUNCTIONALITY{Style.RESET_ALL}\n")

    tester.test_autocomplete("AC: 'tha'", "tha", ["THANK", "THAT", "THAN"])
    tester.test_autocomplete("AC: 'fath'", "fath", ["FATHER"])
    tester.test_autocomplete("AC: 'moth'", "moth", ["MOTHER"])
    tester.test_autocomplete("AC: 'thank you'", "thank you", ["THANK"])

    # ========== ALPHABET & NUMERALS ==========
    print(f"{Fore.YELLOW}🔢 ALPHABET & NUMERALS{Style.RESET_ALL}\n")

    tester.test("Letter: 'A'", "A", "A", 0.99)
    tester.test("Letter: 'Z'", "Z", "Z", 0.99)
    tester.test("Number: '1'", "1", "1", 0.99)
    tester.test("Number: '100'", "100", "100", 0.99)

    # ========== EDGE CASES ==========
    print(f"{Fore.YELLOW}🔬 EDGE CASES{Style.RESET_ALL}\n")

    # Empty and invalid queries
    tester.test_404("Empty query: ''", "")
    tester.test_404("Non-existent: 'xyzabc123'", "xyzabc123")
    tester.test_404("Gibberish: 'qwerty'", "qwerty")

    # Special characters (should be handled gracefully)
    tester.test("Special: 'father!'", "father!", "FATHER", 0.7)
    tester.test("Special: 'hello?'", "hello?", "HELLO", 0.7)

    # Very long queries (should extract main word)
    tester.test("Long: 'I want to thank you very much'",
                "I want to thank you very much", "THANK", 0.8)

    # ========== COMMON USER QUERIES ==========
    print(f"{Fore.YELLOW}👤 COMMON USER QUERIES{Style.RESET_ALL}\n")

    tester.test("User: 'how to sign hello'", "how to sign hello", "HELLO", 0.7)
    tester.test("User: 'sign for father'", "sign for father", "FATHER", 0.8)
    tester.test("User: 'what is the sign for love'",
                "what is the sign for love", "LOVE", 0.7)

    # ========== PRINT SUMMARY ==========
    tester.print_summary()

    # Save results to file
    with open('test_results.json', 'w') as f:
        json.dump({
            'total': tester.passed + tester.failed,
            'passed': tester.passed,
            'failed': tester.failed,
            'success_rate': (tester.passed / (tester.passed + tester.failed) * 100) if (tester.passed + tester.failed) > 0 else 0,
            'results': tester.test_results
        }, f, indent=2)

    print(f"\n📁 Results saved to: test_results.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Tests interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Test suite failed: {e}{Style.RESET_ALL}")
