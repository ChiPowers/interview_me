# YourApp/utils/helper_functions.py

from genai_resume_app.services import vectorstore_service

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os
from langchain.docstore.document import Document
import re

from typing import List, Dict, Tuple, Optional
from langchain.text_splitter import (
    CharacterTextSplitter, 
    RecursiveCharacterTextSplitter,
    TextSplitter
)

from dataclasses import dataclass
from enum import Enum

class AcademicCVSplitter(TextSplitter):
    """
    Specialized splitter for academic CVs with extensive publication lists.
    Preserves complete citations and handles academic formatting.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,  # Larger for academic content
        chunk_overlap: int = 0,   # No overlap for citations
        length_function: callable = len,
        keep_separator: bool = True
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator
        )
        
        # Academic CV sections
        self.academic_sections = [
            "education", "fellowships", "grants", "funding", "awards",
            "research experience", "research", "training", "experience",
            "teaching", "teaching experience", "publications", 
            "peer-reviewed publications", "journal articles",
            "conference presentations", "presentations", "poster presentations",
            "invited talks", "talks", "conferences", "workshops",
            "professional service", "service", "editorial", "reviewing",
            "skills", "technical skills", "languages", "certifications",
            "professional memberships", "memberships", "affiliations"
        ]

    def split_text(self, text: str) -> List[str]:
        """Split academic CV preserving citation integrity."""
        
        # Clean and preprocess
        text = self._preprocess_academic_cv(text)
        
        # Identify sections
        sections = self._extract_academic_sections(text)
        
        chunks = []
        for section_name, section_content in sections.items():
            section_chunks = self._process_academic_section(section_name, section_content)
            chunks.extend(section_chunks)
        
        return [chunk for chunk in chunks if chunk.strip()]

    def _preprocess_academic_cv(self, text: str) -> str:
        """Clean academic CV text while preserving citation structure."""
        
        # Normalize line breaks but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common academic formatting issues
        # Handle author lists that span multiple lines
        text = re.sub(r'([A-Z][a-z]+),\s*\n\s*([A-Z][a-z]+)', r'\1, \2', text)
        
        # Fix split journal names
        text = re.sub(r'([A-Za-z])\.\s*\n\s*([A-Z][a-z])', r'\1. \2', text)
        
        # Handle year patterns that might be split
        text = re.sub(r'\(\s*\n\s*(\d{4})\s*\n\s*\)', r'(\1)', text)
        
        # Clean up excessive spaces but preserve intentional formatting
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

    def _extract_academic_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from academic CV."""
        sections = {}
        current_section = "Header"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Check if this is a section header
            is_section_header = False
            
            # Look for standalone section headers
            if (line_clean and 
                len(line_clean) < 100 and  # Section headers are usually short
                not any(c.islower() for c in line_clean.replace(' ', '')) and  # All caps/title case
                any(section in line_lower for section in self.academic_sections)):
                
                is_section_header = True
                matched_section = next(s for s in self.academic_sections if s in line_lower)
                
            # Also check for common academic section patterns
            elif (line_clean and 
                  re.match(r'^[A-Z][A-Za-z\s&]+$', line_clean) and 
                  len(line_clean.split()) <= 4 and
                  any(keyword in line_lower for keyword in 
                      ['education', 'publications', 'research', 'teaching', 'experience', 'grants', 'awards'])):
                is_section_header = True
                matched_section = line_lower
            
            if is_section_header:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = matched_section.title()
                current_content = []
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def _process_academic_section(self, section_name: str, content: str) -> List[str]:
        """Process each section according to its type."""
        
        section_lower = section_name.lower()
        
        if 'publication' in section_lower or 'presentation' in section_lower:
            return self._split_publications_section(section_name, content)
        elif 'education' in section_lower:
            return self._split_education_section(section_name, content)
        elif 'experience' in section_lower or 'research' in section_lower:
            return self._split_experience_section(section_name, content)
        elif any(keyword in section_lower for keyword in ['grant', 'fellowship', 'award']):
            return self._split_grants_section(section_name, content)
        else:
            return self._split_general_section(section_name, content)

    def _split_publications_section(self, section_name: str, content: str) -> List[str]:
        """Split publications while keeping each citation intact."""
        
        # Add section header to first chunk
        header = f"{section_name}\n" + "="*len(section_name) + "\n"
        
        # Split by publication entries
        publications = self._identify_publications(content)
        
        if not publications:
            # Fallback: split by paragraph if no clear citations found
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            return [header + paragraphs[0]] + paragraphs[1:] if paragraphs else [header + content]
        
        chunks = []
        current_chunk = header
        
        for pub in publications:
            # Check if adding this publication would exceed chunk size
            if len(current_chunk + pub) > self.chunk_size and current_chunk != header:
                # Save current chunk and start new one
                chunks.append(current_chunk.rstrip())
                current_chunk = f"{section_name} (continued)\n" + "="*20 + "\n"
            
            current_chunk += pub + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.rstrip())
        
        return chunks

    def _identify_publications(self, content: str) -> List[str]:
        """Identify individual publications/citations in text."""
        
        publications = []
        lines = content.split('\n')
        current_pub = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if current_pub:
                    publications.append('\n'.join(current_pub))
                    current_pub = []
                continue
            
            # Check if this line starts a new publication
            # Look for author patterns: "LastName, F." or "LastName, F, LastName2, F."
            author_pattern = r'^[A-Z][a-z]+,?\s+[A-Z]\.?(?:,?\s+[A-Z][a-z]+,?\s+[A-Z]\.?)*'
            
            # Or year patterns at start: "(2019)" or "2019."
            year_pattern = r'^(?:\(?\d{4}\)?\.?\s+|.*\(\d{4}\))'
            
            # Or typical academic citation start
            citation_start = (re.match(author_pattern, line) or 
                            re.match(year_pattern, line) or
                            (len(current_pub) == 0))  # First line is always start
            
            if citation_start and current_pub and not line.startswith(' '):
                # This looks like a new citation, save the previous one
                publications.append('\n'.join(current_pub))
                current_pub = [line]
            else:
                # Continue building current citation
                current_pub.append(line)
        
        # Don't forget the last publication
        if current_pub:
            publications.append('\n'.join(current_pub))
        
        return publications

    def _split_education_section(self, section_name: str, content: str) -> List[str]:
        """Split education section by degree entries."""
        
        header = f"{section_name}\n" + "="*len(section_name) + "\n"
        
        # Split by year patterns or degree indicators
        entries = []
        lines = content.split('\n')
        current_entry = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for year/degree patterns that indicate new entries
            if (re.match(r'^\d{4}', line) or 
                re.match(r'^(Ph\.?D|M\.?S\.?|M\.?A\.?|B\.?S\.?|B\.?A\.?)', line)) and current_entry:
                
                entries.append('\n'.join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        # Combine entries into appropriately sized chunks
        return self._combine_entries_to_chunks(header, entries)

    def _split_experience_section(self, section_name: str, content: str) -> List[str]:
        """Split research/work experience by position entries."""
        
        header = f"{section_name}\n" + "="*len(section_name) + "\n"
        
        # Split by date ranges or position titles
        entries = []
        lines = content.split('\n')
        current_entry = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for date ranges or position titles
            date_range_pattern = r'^\d{4}[-–]\d{4}|^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
            position_pattern = r'^[A-Z][a-z].*(?:Scholar|Fellow|Assistant|Associate|Professor|Researcher|Specialist)'
            
            if ((re.match(date_range_pattern, line) or re.search(position_pattern, line)) 
                and current_entry):
                
                entries.append('\n'.join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        return self._combine_entries_to_chunks(header, entries)

    def _split_grants_section(self, section_name: str, content: str) -> List[str]:
        """Split grants/fellowships by individual awards."""
        
        header = f"{section_name}\n" + "="*len(section_name) + "\n"
        
        # Split by year or award patterns
        entries = []
        lines = content.split('\n')
        current_entry = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for year patterns that indicate new grants
            if re.match(r'^\d{4}', line) and current_entry:
                entries.append('\n'.join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        return self._combine_entries_to_chunks(header, entries)

    def _split_general_section(self, section_name: str, content: str) -> List[str]:
        """Split other sections by natural paragraphs."""
        
        header = f"{section_name}\n" + "="*len(section_name) + "\n"
        
        if len(header + content) <= self.chunk_size:
            return [header + content]
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return [header + content]
        
        return self._combine_entries_to_chunks(header, paragraphs)

    def _combine_entries_to_chunks(self, header: str, entries: List[str]) -> List[str]:
        """Combine entries into appropriately sized chunks."""
        
        if not entries:
            return [header]
        
        chunks = []
        current_chunk = header
        
        for entry in entries:
            # Check if adding this entry would exceed chunk size
            potential_chunk = current_chunk + entry + "\n\n"
            
            if len(potential_chunk) > self.chunk_size and current_chunk != header:
                # Save current chunk and start new one
                chunks.append(current_chunk.rstrip())
                section_name = header.split('\n')[0]
                current_chunk = f"{section_name} (continued)\n" + "="*20 + "\n"
            
            current_chunk += entry + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.rstrip())
        
        return chunks

    def create_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        """Create documents with academic-specific metadata."""
        
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # Extract academic metadata
            academic_metadata = self._extract_academic_metadata(text)
            metadata.update(academic_metadata)
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents

    def _extract_academic_metadata(self, text: str) -> Dict[str, str]:
        """Extract academic-specific metadata."""
        
        metadata = {}
        text_lower = text.lower()
        
        # Identify section type
        if any(keyword in text_lower for keyword in ['phd', 'ph.d', 'master', 'bachelor', 'university', 'college']):
            metadata['section_type'] = 'education'
        elif any(keyword in text_lower for keyword in ['journal', 'proceedings', 'conference', 'published']):
            metadata['section_type'] = 'publications'
        elif any(keyword in text_lower for keyword in ['postdoc', 'researcher', 'fellow', 'scholar', 'assistant']):
            metadata['section_type'] = 'research_experience'
        elif any(keyword in text_lower for keyword in ['grant', 'fellowship', 'award', 'funding']):
            metadata['section_type'] = 'funding'
        elif any(keyword in text_lower for keyword in ['teaching', 'instructor', 'course', 'students']):
            metadata['section_type'] = 'teaching'
        elif any(keyword in text_lower for keyword in ['presentation', 'poster', 'conference', 'society']):
            metadata['section_type'] = 'presentations'
        else:
            metadata['section_type'] = 'general'
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            metadata['years'] = list(set(years))
            metadata['year_range'] = f"{min(years)}-{max(years)}" if len(set(years)) > 1 else years[0]
        
        # Count publications in this chunk
        if metadata['section_type'] == 'publications':
            # Rough count based on common citation patterns
            pub_count = len(re.findall(r'\.\s+[A-Z].*?\.\s*$', text, re.MULTILINE))
            metadata['publication_count'] = pub_count
        
        return metadata


class DocumentType(Enum):
    ACADEMIC_CV = "academic_cv"
    RESUME = "resume"
    PROJECT_DESCRIPTION = "project_description"
    PERSONAL_STATEMENT = "personal_statement"
    PORTFOLIO = "portfolio"
    GENERAL_PROFESSIONAL = "general_professional"

@dataclass
class SplitterConfig:
    splitter_type: str
    chunk_size: int
    chunk_overlap: int
    separators: Optional[List[str]] = None
    metadata_enhancer: Optional[callable] = None

class ProfessionalDocumentClassifier:
    """
    Classifies professional/career documents for interview preparation
    """
    
    def __init__(self):
        # Academic CV indicators (PhD, research focus)
        self.academic_cv_indicators = {
            'strong_signals': ['publications', 'research experience', 'dissertation', 
                             'postdoctoral', 'fellowship', 'grants', 'conference presentations',
                             'peer-reviewed', 'journal of', 'proceedings', 'et al'],
            'degree_patterns': [r'\b(Ph\.?D|Doctorate|Postdoc)\b'],
            'citation_patterns': [r'\(\d{4}\)', r'et al\.', r'Journal of', r'Proceedings of'],
            'weight': 0.0
        }
        
        # Resume indicators (industry focus)
        self.resume_indicators = {
            'strong_signals': ['work experience', 'professional experience', 'employment',
                             'achievements', 'responsibilities', 'skills', 'technologies',
                             'certifications', 'languages', 'references'],
            'action_words': ['managed', 'led', 'developed', 'implemented', 'achieved',
                           'increased', 'reduced', 'created', 'designed', 'collaborated'],
            'format_patterns': [r'•', r'-\s', r'\d{4}\s*-\s*\d{4}', r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'],
            'weight': 0.0
        }
        
        # Project description indicators
        self.project_indicators = {
            'strong_signals': ['project', 'github', 'repository', 'demo', 'features',
                             'technologies used', 'architecture', 'implementation',
                             'challenges', 'solutions', 'outcome', 'results'],
            'tech_patterns': [r'\b(Python|JavaScript|React|Django|Flask|SQL|AWS|Docker|Git)\b',
                            r'\b(machine learning|ML|AI|data science|backend|frontend)\b'],
            'project_patterns': [r'github\.com/', r'gitlab\.com/', r'live demo', r'deployed'],
            'weight': 0.0
        }
        
        # Personal statement indicators
        self.personal_statement_indicators = {
            'strong_signals': ['passion', 'motivated', 'career goal', 'aspiration',
                             'why i', 'my journey', 'personal', 'believe', 'vision',
                             'objective', 'summary', 'about me'],
            'personal_patterns': [r'\bI\s+am\b', r'\bMy\s+goal\b', r'\bI\s+believe\b',
                                r'\bpassionate\s+about\b', r'\bmotivated\s+by\b'],
            'weight': 0.0
        }
        
        # Portfolio indicators
        self.portfolio_indicators = {
            'strong_signals': ['portfolio', 'showcase', 'gallery', 'work samples',
                             'case study', 'design process', 'before and after',
                             'client work', 'freelance', 'creative'],
            'creative_patterns': [r'\bdesign\b', r'\bUI/UX\b', r'\bgraphic\b', 
                                r'\bvisual\b', r'\bcreative\b'],
            'weight': 0.0
        }

    def classify_document(self, document: Document) -> Tuple[DocumentType, float]:
        """
        Classify a professional document and return type with confidence score
        """
        content = document.page_content.lower()
        metadata = document.metadata
        
        # Reset weights
        for indicator_set in [self.academic_cv_indicators, self.resume_indicators, 
                             self.project_indicators, self.personal_statement_indicators,
                             self.portfolio_indicators]:
            indicator_set['weight'] = 0.0
        
        # Check filename/source hints
        source_hints = self._get_source_hints(metadata)
        
        # Analyze content
        self._analyze_content(content)
        self._apply_source_hints(source_hints)
        
        # Calculate scores
        scores = {
            DocumentType.ACADEMIC_CV: self.academic_cv_indicators['weight'],
            DocumentType.RESUME: self.resume_indicators['weight'],
            DocumentType.PROJECT_DESCRIPTION: self.project_indicators['weight'],
            DocumentType.PERSONAL_STATEMENT: self.personal_statement_indicators['weight'],
            DocumentType.PORTFOLIO: self.portfolio_indicators['weight'],
            DocumentType.GENERAL_PROFESSIONAL: 1.0  # Baseline
        }
        
        # Find best match
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # If no strong indicators, classify as general professional
        if confidence < 2.0:
            return DocumentType.GENERAL_PROFESSIONAL, confidence
        
        return best_type, confidence

    def _get_source_hints(self, metadata: Dict) -> Dict[str, float]:
        """Extract hints from metadata (filename, source, etc.)"""
        hints = {}
        
        source = metadata.get('source', '').lower()
        
        if any(term in source for term in ['cv', 'vitae', 'curriculum']):
            hints['academic_cv'] = 3.0
        elif any(term in source for term in ['resume']):
            hints['resume'] = 3.0
        elif any(term in source for term in ['project', 'portfolio', 'work']):
            hints['project'] = 2.0
        elif any(term in source for term in ['statement', 'cover', 'letter', 'about']):
            hints['personal_statement'] = 2.0
        elif any(term in source for term in ['showcase', 'gallery', 'samples']):
            hints['portfolio'] = 2.0
            
        return hints

    def _analyze_content(self, content: str):
        """Analyze content for classification signals"""
        
        # Academic CV analysis
        academic_score = 0
        for signal in self.academic_cv_indicators['strong_signals']:
            if signal in content:
                academic_score += 1.0
        
        for pattern in self.academic_cv_indicators['degree_patterns']:
            academic_score += len(re.findall(pattern, content, re.IGNORECASE)) * 2.0
            
        for pattern in self.academic_cv_indicators['citation_patterns']:
            academic_score += len(re.findall(pattern, content, re.IGNORECASE)) * 0.5
        
        self.academic_cv_indicators['weight'] = academic_score
        
        # Resume analysis
        resume_score = 0
        for signal in self.resume_indicators['strong_signals']:
            if signal in content:
                resume_score += 0.8
        
        for word in self.resume_indicators['action_words']:
            if word in content:
                resume_score += 0.5
                
        for pattern in self.resume_indicators['format_patterns']:
            resume_score += len(re.findall(pattern, content, re.IGNORECASE)) * 0.3
        
        self.resume_indicators['weight'] = resume_score
        
        # Project analysis
        project_score = 0
        for signal in self.project_indicators['strong_signals']:
            if signal in content:
                project_score += 1.0
                
        for pattern in self.project_indicators['tech_patterns']:
            project_score += len(re.findall(pattern, content, re.IGNORECASE)) * 0.5
            
        for pattern in self.project_indicators['project_patterns']:
            project_score += len(re.findall(pattern, content, re.IGNORECASE)) * 1.0
        
        self.project_indicators['weight'] = project_score
        
        # Personal statement analysis
        personal_score = 0
        for signal in self.personal_statement_indicators['strong_signals']:
            if signal in content:
                personal_score += 1.0
                
        for pattern in self.personal_statement_indicators['personal_patterns']:
            personal_score += len(re.findall(pattern, content, re.IGNORECASE)) * 1.0
        
        self.personal_statement_indicators['weight'] = personal_score
        
        # Portfolio analysis
        portfolio_score = 0
        for signal in self.portfolio_indicators['strong_signals']:
            if signal in content:
                portfolio_score += 1.0
                
        for pattern in self.portfolio_indicators['creative_patterns']:
            portfolio_score += len(re.findall(pattern, content, re.IGNORECASE)) * 0.5
        
        self.portfolio_indicators['weight'] = portfolio_score

    def _apply_source_hints(self, hints: Dict[str, float]):
        """Apply source-based hints to weights"""
        
        if 'academic_cv' in hints:
            self.academic_cv_indicators['weight'] += hints['academic_cv']
        if 'resume' in hints:
            self.resume_indicators['weight'] += hints['resume']
        if 'project' in hints:
            self.project_indicators['weight'] += hints['project']
        if 'personal_statement' in hints:
            self.personal_statement_indicators['weight'] += hints['personal_statement']
        if 'portfolio' in hints:
            self.portfolio_indicators['weight'] += hints['portfolio']


class ProfessionalSplitterAgent:
    """
    Streamlined agent for professional/career documents
    """
    
    def __init__(self):
        self.classifier = ProfessionalDocumentClassifier()
        self.splitter_configs = self._initialize_splitter_configs()
        
    def _initialize_splitter_configs(self) -> Dict[DocumentType, SplitterConfig]:
        """Initialize splitter configurations for professional documents"""
        
        return {
            DocumentType.ACADEMIC_CV: SplitterConfig(
                splitter_type="academic_cv",
                chunk_size=1200,
                chunk_overlap=0,
                metadata_enhancer=self._enhance_academic_metadata
            ),
            
            DocumentType.RESUME: SplitterConfig(
                splitter_type="recursive",
                chunk_size=800,
                chunk_overlap=50,
                separators=["\n\n", "\n•", "\n-", "\n", ". ", " "],
                metadata_enhancer=self._enhance_resume_metadata
            ),
            
            DocumentType.PROJECT_DESCRIPTION: SplitterConfig(
                splitter_type="recursive",
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n## ", "\n### ", "\n**", "\n\n", "\n", ". ", " "],
                metadata_enhancer=self._enhance_project_metadata
            ),
            
            DocumentType.PERSONAL_STATEMENT: SplitterConfig(
                splitter_type="recursive",
                chunk_size=600,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " "],
                metadata_enhancer=self._enhance_personal_metadata
            ),
            
            DocumentType.PORTFOLIO: SplitterConfig(
                splitter_type="recursive",
                chunk_size=900,
                chunk_overlap=75,
                separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
                metadata_enhancer=self._enhance_portfolio_metadata
            ),
            
            DocumentType.GENERAL_PROFESSIONAL: SplitterConfig(
                splitter_type="character",
                chunk_size=1000,
                chunk_overlap=20,
                separators=["\n\n"],
                metadata_enhancer=None
            )
        }

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Main method: automatically classify and split professional documents
        """
        all_chunks = []
        classification_report = []
        
        for i, doc in enumerate(documents):
            # Classify document
            doc_type, confidence = self.classifier.classify_document(doc)
            
            classification_report.append({
                'doc_index': i,
                'doc_type': doc_type.value,
                'confidence': confidence,
                'source': doc.metadata.get('source', 'unknown')
            })
            
            # Get splitter config
            config = self.splitter_configs[doc_type]
            
            # Split document
            chunks = self._split_single_document(doc, config, doc_type)
            
            # Enhance metadata
            if config.metadata_enhancer:
                chunks = config.metadata_enhancer(chunks, doc_type)
            
            all_chunks.extend(chunks)
        
        # Print classification report
        self._print_classification_report(classification_report)
        
        return all_chunks

    def _split_single_document(self, doc: Document, config: SplitterConfig, doc_type: DocumentType) -> List[Document]:
        """Split a single document using the specified configuration"""
        
        if config.splitter_type == "academic_cv":
            # Use your custom academic splitter
            splitter = AcademicCVSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            
        elif config.splitter_type == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                separators=config.separators,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len
            )
            
        elif config.splitter_type == "character":
            splitter = CharacterTextSplitter(
                separator=config.separators[0] if config.separators else "\n\n",
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len
            )
        
        # Split the document
        chunks = splitter.split_documents([doc])
        
        # Add document type to metadata
        for chunk in chunks:
            chunk.metadata['document_type'] = doc_type.value
            chunk.metadata['splitter_used'] = config.splitter_type
            
        return chunks

    def _print_classification_report(self, report: List[Dict]):
        """Print classification results"""
        print("\n" + "="*50)
        print("PROFESSIONAL DOCUMENT CLASSIFICATION")
        print("="*50)
        
        for item in report:
            print(f"Doc {item['doc_index']}: {item['doc_type']} "
                  f"(confidence: {item['confidence']:.1f}) - {item['source']}")
        
        print("="*50 + "\n")

    # Metadata enhancer methods
    def _enhance_academic_metadata(self, chunks: List[Document], doc_type: DocumentType) -> List[Document]:
        """Add academic CV specific metadata (handled by AcademicCVSplitter)"""
        return chunks

    def _enhance_resume_metadata(self, chunks: List[Document], doc_type: DocumentType) -> List[Document]:
        """Add resume specific metadata"""
        for chunk in chunks:
            content = chunk.page_content.lower()
            
            if any(word in content for word in ['experience', 'employment', 'work history']):
                chunk.metadata['section_type'] = 'experience'
            elif any(word in content for word in ['education', 'degree', 'university', 'college']):
                chunk.metadata['section_type'] = 'education'
            elif any(word in content for word in ['skills', 'technologies', 'programming', 'software']):
                chunk.metadata['section_type'] = 'skills'
            elif any(word in content for word in ['certifications', 'licenses', 'credentials']):
                chunk.metadata['section_type'] = 'certifications'
            elif any(word in content for word in ['projects', 'portfolio', 'work samples']):
                chunk.metadata['section_type'] = 'projects'
            elif any(word in content for word in ['summary', 'objective', 'profile']):
                chunk.metadata['section_type'] = 'summary'
            else:
                chunk.metadata['section_type'] = 'general'
            
            # Extract years for experience dating
            years = re.findall(r'\b(19|20)\d{2}\b', chunk.page_content)
            if years:
                chunk.metadata['years'] = list(set(years))
                chunk.metadata['year_range'] = f"{min(years)}-{max(years)}" if len(set(years)) > 1 else years[0]
        
        return chunks

    def _enhance_project_metadata(self, chunks: List[Document], doc_type: DocumentType) -> List[Document]:
        """Add project specific metadata"""
        for chunk in chunks:
            content = chunk.page_content.lower()
            
            # Extract technologies
            tech_patterns = [
                r'\b(python|javascript|react|vue|angular|node|django|flask|fastapi)\b',
                r'\b(sql|postgresql|mysql|mongodb|redis)\b',
                r'\b(aws|azure|gcp|docker|kubernetes|git)\b',
                r'\b(machine learning|ml|ai|data science|nlp)\b'
            ]
            
            technologies = set()
            for pattern in tech_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                technologies.update(matches)
            
            if technologies:
                chunk.metadata['technologies'] = list(technologies)
            
            # Identify project phases
            if any(word in content for word in ['planning', 'design', 'architecture', 'requirements']):
                chunk.metadata['project_phase'] = 'planning'
            elif any(word in content for word in ['implementation', 'development', 'coding', 'built']):
                chunk.metadata['project_phase'] = 'development'
            elif any(word in content for word in ['testing', 'debugging', 'validation']):
                chunk.metadata['project_phase'] = 'testing'
            elif any(word in content for word in ['deployment', 'launch', 'production', 'live']):
                chunk.metadata['project_phase'] = 'deployment'
            elif any(word in content for word in ['results', 'outcome', 'impact', 'metrics']):
                chunk.metadata['project_phase'] = 'results'
        
        return chunks

    def _enhance_personal_metadata(self, chunks: List[Document], doc_type: DocumentType) -> List[Document]:
        """Add personal statement specific metadata"""
        for chunk in chunks:
            content = chunk.page_content.lower()
            
            if any(word in content for word in ['goal', 'aspiration', 'objective', 'aim']):
                chunk.metadata['statement_type'] = 'goals'
            elif any(word in content for word in ['experience', 'background', 'journey']):
                chunk.metadata['statement_type'] = 'background'
            elif any(word in content for word in ['passion', 'interest', 'motivated', 'love']):
                chunk.metadata['statement_type'] = 'motivation'
            elif any(word in content for word in ['vision', 'future', 'plan', 'career']):
                chunk.metadata['statement_type'] = 'vision'
            else:
                chunk.metadata['statement_type'] = 'general'
        
        return chunks

    def _enhance_portfolio_metadata(self, chunks: List[Document], doc_type: DocumentType) -> List[Document]:
        """Add portfolio specific metadata"""
        for chunk in chunks:
            content = chunk.page_content.lower()
            
            if any(word in content for word in ['case study', 'project', 'work sample']):
                chunk.metadata['portfolio_type'] = 'case_study'
            elif any(word in content for word in ['design', 'ui', 'ux', 'visual', 'graphic']):
                chunk.metadata['portfolio_type'] = 'design'
            elif any(word in content for word in ['code', 'programming', 'development', 'technical']):
                chunk.metadata['portfolio_type'] = 'technical'
            elif any(word in content for word in ['writing', 'content', 'copy', 'blog']):
                chunk.metadata['portfolio_type'] = 'writing'
            else:
                chunk.metadata['portfolio_type'] = 'general'
        
        return chunks

def load_docs(path_to_pdfs):
    loader = PyPDFDirectoryLoader(path_to_pdfs)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Enhanced professional document splitter - now with automatic classification!
    """
    agent = ProfessionalSplitterAgent()
    return agent.split_documents(documents)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    template = """
    You are Chivon E. Powers interviewing for Senior Data Scientist, AI Engineer, and Machine Learning Engineering roles.\
        Use the following context to answer interview questions in a way that describes how your experience \
        and skills relate to the requirements for these types of jobs. Provide conversational responses in the first person up to 3 sentences.\

        Context: {context} \
        Interview Question: \
        {question} \
        Answer:
        """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def session_first_embed_and_store(doc_path=os.environ.get("rag_pdf_path"),
                                  db_path=os.environ.get("faiss_index_path")):
    texts = load_docs(doc_path)
    chunks = split_docs(texts)
    vectorstore_service.embed_chunks_and_upload_to_faiss(chunks, db_path)

 
