# Frontend Migration Analysis: Streamlit vs Alternatives

**Document Version:** 1.0
**Date:** January 2026
**Project:** RESA (Rocket Engine Sizing & Analysis)

---

## Executive Summary

This document analyzes the tradeoffs of migrating RESA's frontend from Streamlit to alternative frameworks (React, Qt, Vue.js, or others). The analysis considers migration effort, long-term maintainability, feature extensibility, and alignment with project goals.

**Key Findings:**
- Current Streamlit implementation: ~3,300 LOC across 12 pages
- Migration to React: 3-6 months estimated effort, requires API layer development
- Migration to Qt: 2-4 months estimated effort, native desktop focus
- Recommendation: **Incremental enhancement of Streamlit** for near-term, with **React migration** as a strategic long-term option if web distribution or advanced UI becomes critical

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Frontend Alternatives Evaluated](#2-frontend-alternatives-evaluated)
3. [Detailed Comparison Matrix](#3-detailed-comparison-matrix)
4. [Migration Effort Estimation](#4-migration-effort-estimation)
5. [Advantages and Disadvantages](#5-advantages-and-disadvantages)
6. [Future Maintainability Analysis](#6-future-maintainability-analysis)
7. [Feature Extensibility Assessment](#7-feature-extensibility-assessment)
8. [Recommendations](#8-recommendations)
9. [Appendix: Implementation Roadmaps](#9-appendix-implementation-roadmaps)

---

## 1. Current State Analysis

### 1.1 Codebase Metrics

| Metric | Value |
|--------|-------|
| Total UI Lines of Code | 3,281 |
| Number of Pages | 12 |
| Plotly Visualizations | 63+ instances |
| Interactive Elements | 270+ components |
| Backend Module Imports | 19 modules |
| Session State Variables | 21 keys |

### 1.2 Page Complexity Distribution

| Complexity | Pages | Lines |
|------------|-------|-------|
| High (>300 LOC) | 5 | n2o_cooling (746), tank (490), contour (422), projects (383), design (331) |
| Medium (100-300 LOC) | 3 | igniter (115), monte_carlo (113), injector (111) |
| Low (<100 LOC) | 4 | throttle (68), optimization (48), analysis (39) |

### 1.3 Current Architecture Strengths

1. **Rapid Development**: Streamlit enables fast prototyping with minimal frontend code
2. **Python-Native**: No language context switching; entire stack is Python
3. **Plotly Integration**: Seamless embedding of interactive visualizations
4. **Low Barrier**: Aerospace engineers can contribute without web development expertise
5. **Single Deployment**: One process serves both backend and frontend

### 1.4 Current Architecture Limitations

1. **Session State Fragility**: State lost on page refresh, no persistence
2. **Limited UI Customization**: Constrained layout system, no pixel-level control
3. **Performance Ceiling**: Full page reruns on every interaction
4. **No Offline Mode**: Requires running server, no standalone executable
5. **Scalability Limits**: Single-threaded, difficult to scale horizontally
6. **Testing Challenges**: UI components tightly coupled to Streamlit

### 1.5 Backend Integration Pattern

```
Current Architecture:
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Direct Python imports from resa package        │    │
│  │  - resa.core (Engine, EngineConfig)            │    │
│  │  - resa.solvers (CEASolver, CoolingSolver)     │    │
│  │  - resa.visualization (Plotters)               │    │
│  │  - resa.addons (Igniter, Injector, Tank)       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Direct method calls
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Python Backend                         │
│  - Physics calculations (pure functions)                 │
│  - Solver orchestration                                  │
│  - Result dataclasses (frozen=True)                     │
└─────────────────────────────────────────────────────────┘
```

**Critical Coupling Point**: The `AnalysisModule` interface in `core/interfaces.py` (lines 289-333) explicitly accepts Streamlit's `st` object, creating a hard dependency on Streamlit for addon modules.

---

## 2. Frontend Alternatives Evaluated

### 2.1 React + FastAPI

**Architecture:**
```
┌──────────────────┐    HTTP/WebSocket    ┌──────────────────┐
│   React SPA      │ ◄──────────────────► │   FastAPI        │
│   (TypeScript)   │        JSON          │   (Python)       │
└──────────────────┘                      └──────────────────┘
```

**Technology Stack:**
- React 18+ with TypeScript
- Vite or Next.js for build tooling
- React Query for server state
- Zustand or Redux for client state
- Plotly.js or Recharts for visualization
- FastAPI backend with Pydantic models

### 2.2 Qt (PySide6/PyQt6)

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                    Qt Desktop App                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  QML/Widgets UI layer                           │    │
│  │           │                                     │    │
│  │           │ Direct Python calls                 │    │
│  │           ▼                                     │    │
│  │  Python backend (resa package)                  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Technology Stack:**
- PySide6 (LGPL) or PyQt6 (GPL/Commercial)
- QML for modern UI or Qt Widgets for traditional
- PyQtGraph or embedded Plotly for visualization
- cx_Freeze or PyInstaller for distribution

### 2.3 Vue.js + Flask

**Architecture:** Similar to React, with Vue as the frontend framework.

**Technology Stack:**
- Vue 3 with Composition API
- Vuetify or Quasar for UI components
- Pinia for state management
- Flask or FastAPI backend

### 2.4 Electron + React

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                    Electron App                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Chromium (React UI)                            │    │
│  │           │                                     │    │
│  │           │ IPC                                 │    │
│  │           ▼                                     │    │
│  │  Node.js main process ◄────► Python subprocess  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 2.5 Enhanced Streamlit

**Architecture:** Keep Streamlit, add improvements:
- REST API layer for headless automation
- Better state management patterns
- Component library extraction
- Caching optimization

---

## 3. Detailed Comparison Matrix

### 3.1 Technical Capabilities

| Capability | Streamlit | React | Qt | Vue |
|------------|-----------|-------|-----|-----|
| Custom UI layouts | Limited | Full | Full | Full |
| Pixel-level control | No | Yes | Yes | Yes |
| Responsive design | Basic | Excellent | Manual | Excellent |
| Offline operation | No | Possible | Yes | Possible |
| Native desktop | No | Via Electron | Yes | Via Electron |
| Mobile support | Limited | Yes | Limited | Yes |
| Real-time updates | WebSocket | WebSocket | Native | WebSocket |
| 3D visualization | Plotly only | WebGL/Three.js | OpenGL/Plotly | WebGL/Three.js |
| Keyboard shortcuts | Limited | Full | Full | Full |
| Drag-and-drop | No | Yes | Yes | Yes |
| Multi-window | No | Possible | Yes | Possible |
| Theming | Basic | Full | Full | Full |

### 3.2 Development Characteristics

| Characteristic | Streamlit | React | Qt | Vue |
|----------------|-----------|-------|-----|-----|
| Learning curve | Low | Medium-High | Medium | Medium |
| Python-only dev | Yes | No | Yes | No |
| Hot reload | Yes | Yes | Partial | Yes |
| Component reuse | Limited | Excellent | Good | Excellent |
| Testing tools | Limited | Excellent | Good | Excellent |
| Type safety | Optional | TypeScript | Python typing | TypeScript |
| Bundle size | N/A (server) | ~200KB+ | ~50MB+ | ~150KB+ |
| Build complexity | None | Moderate | Low | Moderate |

### 3.3 Operational Characteristics

| Characteristic | Streamlit | React | Qt | Vue |
|----------------|-----------|-------|-----|-----|
| Deployment model | Server | Static/Server | Desktop | Static/Server |
| Scaling | Difficult | Easy | N/A | Easy |
| Multi-user | Problematic | Native | Per-install | Native |
| Authentication | Manual | Libraries | Manual | Libraries |
| API versioning | N/A | Required | N/A | Required |
| CI/CD complexity | Low | Medium | Medium | Medium |

---

## 4. Migration Effort Estimation

### 4.1 React Migration

#### Phase 1: API Layer Development (4-6 weeks)
- Design RESTful API endpoints for all backend operations
- Implement FastAPI routes with Pydantic models
- Create WebSocket endpoints for long-running operations
- Add authentication/session management
- Write API documentation (OpenAPI/Swagger)

**Endpoints Required:**
```
POST   /api/engine/design          # Run engine design
GET    /api/engine/config/validate # Validate configuration
POST   /api/analysis/monte-carlo   # Run Monte Carlo
POST   /api/analysis/throttle      # Generate throttle map
POST   /api/igniter/design         # Design igniter
POST   /api/injector/design        # Design injector
POST   /api/contour/generate       # Generate 3D contour
POST   /api/tank/simulate          # Run tank simulation
GET    /api/projects               # List projects
POST   /api/projects/{id}/version  # Save version
GET    /api/export/{format}        # Export results
WebSocket /ws/progress             # Real-time progress
```

#### Phase 2: React Application Development (8-12 weeks)
- Set up React project with TypeScript
- Implement component library (forms, charts, layouts)
- Recreate 12 pages as React components
- Integrate Plotly.js for visualizations
- Implement client-side state management
- Add routing and navigation

**Component Mapping:**
| Streamlit Page | React Components |
|----------------|------------------|
| design_page.py | EngineConfigForm, DesignRunner, ResultsDashboard, ReportViewer |
| n2o_cooling_page.py | CoolingAnalysisForm, CoolingResults, ParametricStudy |
| contour_page.py | NozzleDesigner, ChannelDesigner, STLExporter, 3DViewer |
| tank_page.py | TankConfigPanel, SimulationRunner, TimeSeriesPlots |
| monte_carlo_page.py | DistributionSetup, MCRunner, HistogramPlots, SensitivityChart |
| projects_page.py | ProjectList, VersionHistory, ComparisonView |
| (shared) | MetricCard, PlotContainer, FormSection, ExportButton |

#### Phase 3: Testing and Refinement (3-4 weeks)
- End-to-end testing
- Performance optimization
- Cross-browser testing
- Accessibility audit
- Documentation

**Total React Migration: 15-22 weeks (3-5 months)**

### 4.2 Qt Migration

#### Phase 1: Application Architecture (2-3 weeks)
- Design Qt application structure
- Set up PySide6 project with QML
- Create main window and navigation
- Design signal/slot architecture

#### Phase 2: UI Development (6-10 weeks)
- Create QML components for all pages
- Integrate PyQtGraph or Plotly for visualization
- Implement forms and input validation
- Add threading for long operations
- Create settings/preferences system

#### Phase 3: Packaging and Distribution (2-3 weeks)
- Set up PyInstaller or cx_Freeze
- Create installers for Windows/macOS/Linux
- Handle dependency bundling (especially RocketCEA, CoolProp)
- Code signing (optional)

**Total Qt Migration: 10-16 weeks (2-4 months)**

### 4.3 Enhanced Streamlit (Recommended Near-Term)

#### Improvements (4-6 weeks)
- Extract reusable components
- Implement proper caching with `@st.cache_data`
- Add REST API layer (FastAPI) alongside Streamlit
- Improve error handling and user feedback
- Add input validation helpers
- Create unit tests for page logic

**Total Enhancement: 4-6 weeks (1-1.5 months)**

---

## 5. Advantages and Disadvantages

### 5.1 React + FastAPI

**Advantages:**
1. **Industry Standard**: Vast ecosystem, abundant talent pool, extensive documentation
2. **Maximum Flexibility**: Complete control over UI/UX, pixel-perfect designs
3. **Scalability**: Stateless API enables horizontal scaling
4. **Performance**: Virtual DOM, code splitting, lazy loading
5. **Testing**: Jest, React Testing Library, Cypress provide excellent coverage
6. **Multi-Platform**: Same codebase can serve web, mobile (React Native), desktop (Electron)
7. **Component Ecosystem**: Thousands of pre-built components (Ant Design, MUI, etc.)
8. **SEO Possible**: With Next.js for server-side rendering (if needed)
9. **Team Collaboration**: Clear separation of frontend/backend enables parallel development
10. **Modern Tooling**: TypeScript, ESLint, Prettier, Storybook

**Disadvantages:**
1. **Development Overhead**: Requires maintaining two codebases (Python + TypeScript)
2. **Skill Requirements**: Team needs JavaScript/TypeScript expertise
3. **API Maintenance**: Must keep frontend and backend API contracts in sync
4. **Build Complexity**: Webpack/Vite configuration, deployment pipelines
5. **Initial Investment**: Significant upfront effort before feature parity
6. **Context Switching**: Engineers must work across language boundaries
7. **Serialization Overhead**: All data must be JSON-serializable
8. **Plotly Integration**: Plotly.js API differs slightly from Python Plotly

### 5.2 Qt (PySide6)

**Advantages:**
1. **Python-Native**: No language switching, entire application in Python
2. **Desktop Performance**: Native widgets, excellent responsiveness
3. **Offline First**: No server required, fully functional without network
4. **Single Executable**: Can distribute as standalone installer
5. **Rich Widgets**: Professional UI components out of the box
6. **OpenGL Integration**: Native 3D rendering without browser limitations
7. **Mature Framework**: 25+ years of development, stable and well-documented
8. **Cross-Platform**: One codebase for Windows, macOS, Linux
9. **Direct Backend Access**: No serialization, direct Python object passing
10. **IP Protection**: Compiled executable harder to reverse-engineer

**Disadvantages:**
1. **No Web Deployment**: Cannot serve users via browser without rewrite
2. **Distribution Burden**: Must build and test installers for each platform
3. **Update Mechanism**: Must implement own update system or rely on reinstalls
4. **Bundle Size**: Executables are large (50-200MB) due to Qt libraries
5. **Limited Component Ecosystem**: Fewer pre-built components than web frameworks
6. **QML Learning Curve**: QML is a separate language to learn (unless using Widgets)
7. **Dependency Bundling**: Complex native dependencies (RocketCEA) are difficult to package
8. **No Concurrent Users**: Each user needs separate installation
9. **Remote Access Difficult**: Cannot easily access from other devices
10. **Licensing Considerations**: PyQt6 is GPL (commercial license required for proprietary)

### 5.3 Vue.js + Flask

**Advantages:**
1. **Gentle Learning Curve**: Easier than React for Python developers
2. **Template Syntax**: HTML-like templates feel familiar
3. **Single-File Components**: Clean organization of template/script/style
4. **Smaller Bundle**: Vue is lighter than React
5. **Progressive Adoption**: Can be added incrementally

**Disadvantages:**
1. **Smaller Ecosystem**: Fewer components and tools than React
2. **Smaller Talent Pool**: Less common than React in job market
3. **TypeScript Support**: Good but not as mature as React's
4. **Same Drawbacks as React**: API layer, build system, etc.

### 5.4 Electron + React

**Advantages:**
1. **Best of Both Worlds**: Web technologies with desktop distribution
2. **Offline Capability**: Works without internet connection
3. **Native Features**: File system access, system tray, notifications
4. **Familiar to Web Devs**: Standard React development experience

**Disadvantages:**
1. **Resource Heavy**: Bundles entire Chromium (~100-200MB)
2. **Memory Usage**: Higher RAM consumption than native apps
3. **Startup Time**: Slower launch than native applications
4. **Complex Architecture**: Node.js + Chromium + Python subprocess
5. **Security Surface**: Chromium vulnerabilities affect app

### 5.5 Enhanced Streamlit (Status Quo+)

**Advantages:**
1. **Minimal Disruption**: No migration, evolutionary improvement
2. **Fastest Time-to-Value**: Improvements available immediately
3. **Team Continuity**: No new skills required
4. **Risk Mitigation**: No risk of failed migration
5. **API Addition**: Can add REST API without replacing Streamlit
6. **Reversible**: Improvements don't preclude future migration

**Disadvantages:**
1. **Ceiling Remains**: Fundamental limitations persist
2. **Technical Debt**: May accumulate workarounds
3. **Opportunity Cost**: Time spent on workarounds vs. proper solution
4. **User Experience**: Cannot achieve modern web app feel
5. **Scaling Issues**: Multi-user scenarios remain problematic

---

## 6. Future Maintainability Analysis

### 6.1 Codebase Health Metrics

| Metric | Streamlit | React | Qt |
|--------|-----------|-------|-----|
| Testability | Poor | Excellent | Good |
| Type Safety | Optional | Enforced | Optional |
| Refactoring Safety | Low | High | Medium |
| Documentation Generation | Manual | Automatic | Manual |
| Dependency Management | pip | npm + pip | pip |
| Code Splitting | No | Yes | Manual |
| Dead Code Detection | Manual | Tooling | Manual |

### 6.2 Long-term Maintenance Considerations

**Streamlit:**
- Dependent on Streamlit Inc.'s roadmap
- Breaking changes between versions possible
- Community components may become unmaintained
- Limited control over performance optimization

**React:**
- Meta's backing ensures long-term support
- Ecosystem stability with clear deprecation paths
- Large community maintains components
- TypeScript provides refactoring confidence
- Well-established patterns for large codebases

**Qt:**
- Qt Company provides long-term support
- Stable API with backward compatibility
- Smaller community but professional-grade
- Well-suited for desktop engineering tools
- Potential licensing changes (has happened before)

### 6.3 Team Scaling Considerations

| Team Size | Streamlit | React | Qt |
|-----------|-----------|-------|-----|
| 1-2 devs | Ideal | Overhead | Good |
| 3-5 devs | Workable | Good | Good |
| 5+ devs | Problematic | Ideal | Good |

**Rationale:**
- Streamlit's single-file pages create merge conflicts with multiple developers
- React's component architecture enables parallel development
- Qt's signal/slot pattern works well for medium teams

---

## 7. Feature Extensibility Assessment

### 7.1 Planned Features and Framework Fit

| Future Feature | Streamlit | React | Qt |
|----------------|-----------|-------|-----|
| Real-time simulation monitoring | Difficult | Native | Native |
| Collaborative editing | Very Difficult | Feasible | Very Difficult |
| Offline mode | Impossible | Electron | Native |
| Mobile companion app | No | React Native | Limited |
| Plugin architecture | Limited | Excellent | Good |
| Custom visualization widgets | Limited | Full | Full |
| Drag-drop workflow builder | Very Limited | Excellent | Good |
| Multi-language support (i18n) | Manual | Libraries | Libraries |
| Keyboard shortcut system | Limited | Full | Full |
| Undo/redo system | Manual | Libraries | Built-in |
| Print/PDF export | Browser | Libraries | Native |
| System integration (file associations) | No | Electron | Native |

### 7.2 Specific RESA Feature Roadmap Assessment

**Planned/Potential Features:**

1. **Real-time Engine Simulation Dashboard**
   - Streamlit: Requires websocket workarounds, limited refresh rates
   - React: Native WebSocket support, optimistic updates, smooth animations
   - Qt: Excellent with QTimer and signal/slot, smooth animations

2. **3D CAD-like Geometry Editor**
   - Streamlit: Plotly only, no true CAD manipulation
   - React: Three.js integration, CAD-style controls possible
   - Qt: OpenGL/QML3D, professional CAD tools exist (FreeCAD uses Qt)

3. **Batch Processing / Automation API**
   - Streamlit: Need separate API anyway
   - React: API is the architecture
   - Qt: Need separate API for headless use

4. **Multi-Engine Project Comparison**
   - Streamlit: Session state makes this complex
   - React: Natural with proper state management
   - Qt: Natural with document-based architecture

5. **Report Customization / Template Editor**
   - Streamlit: Very limited
   - React: Rich text editors, template builders available
   - Qt: Qt has rich text editing widgets

---

## 8. Recommendations

### 8.1 Decision Framework

```
                              ┌──────────────────────────────────┐
                              │    Primary Distribution Mode?     │
                              └──────────────────────────────────┘
                                     │                    │
                               ┌─────┴─────┐        ┌─────┴─────┐
                               │   Web     │        │  Desktop  │
                               └───────────┘        └───────────┘
                                     │                    │
                     ┌───────────────┴───────────────┐    │
                     │  Need advanced UI features?    │    │
                     └───────────────────────────────┘    │
                           │                    │         │
                      ┌────┴────┐          ┌────┴────┐    │
                      │   Yes   │          │   No    │    │
                      └─────────┘          └─────────┘    │
                           │                    │         │
                           ▼                    ▼         ▼
                    ┌──────────────┐    ┌──────────────┐ ┌──────────────┐
                    │    React     │    │  Enhanced    │ │     Qt       │
                    │   Migration  │    │  Streamlit   │ │  Migration   │
                    └──────────────┘    └──────────────┘ └──────────────┘
```

### 8.2 Recommendation by Use Case

| Use Case | Recommended Approach |
|----------|---------------------|
| Internal tool, small team, rapid iteration | Enhanced Streamlit |
| Product for external users, web-based | React + FastAPI |
| Desktop software for engineers, offline use | Qt (PySide6) |
| Maximum reach (web + desktop) | React + Electron |
| API-first with UI as secondary | FastAPI + Streamlit (parallel) |

### 8.3 Recommended Strategy for RESA

**Phase 1: Immediate (1-2 months)**
- Enhance current Streamlit implementation
- Extract reusable components
- Add comprehensive caching
- Implement REST API alongside Streamlit (FastAPI)
- This API enables: CI/CD integration, headless batch processing, future frontend flexibility

**Phase 2: Evaluation (Month 3)**
- Assess whether enhanced Streamlit meets user needs
- Gather user feedback on pain points
- Prototype critical features in React (if needed)

**Phase 3: Migration Decision (Month 4+)**
- If Streamlit limitations are acceptable: Continue with Streamlit + API
- If web UI is critical: Begin React migration
- If desktop distribution is critical: Begin Qt migration
- Hybrid: React for web, keep Streamlit for internal rapid prototyping

### 8.4 Cost-Benefit Summary

| Approach | Effort | Risk | Benefit | Recommendation |
|----------|--------|------|---------|----------------|
| Do Nothing | None | Low | None | Not Recommended |
| Enhanced Streamlit | Low (4-6 weeks) | Very Low | Moderate | **Recommended (Phase 1)** |
| React Migration | High (15-22 weeks) | Medium | High | Conditional (if web UX critical) |
| Qt Migration | Medium (10-16 weeks) | Medium | High | Conditional (if desktop critical) |
| Vue Migration | High (14-20 weeks) | Medium | Medium | Not Recommended |
| Electron | Very High (20+ weeks) | High | Medium | Not Recommended |

---

## 9. Appendix: Implementation Roadmaps

### 9.1 Enhanced Streamlit Roadmap

```
Week 1-2: Architecture Improvements
├── Create component library (resa/ui/components/)
│   ├── metric_card.py
│   ├── parameter_form.py
│   ├── plot_container.py
│   └── export_button.py
├── Implement state management patterns
│   └── state_manager.py (centralized session state)
└── Add caching decorators to expensive operations

Week 3-4: API Layer
├── Create FastAPI application (resa/api/)
│   ├── app.py (main FastAPI app)
│   ├── routes/
│   │   ├── engine.py
│   │   ├── analysis.py
│   │   └── export.py
│   └── models/ (Pydantic request/response models)
└── Document API with OpenAPI spec

Week 5-6: Testing and Documentation
├── Add unit tests for UI components
├── Add integration tests for API
├── Performance profiling and optimization
└── Update documentation
```

### 9.2 React Migration Roadmap

```
Phase 1: Foundation (Weeks 1-6)
├── Week 1-2: FastAPI backend setup
│   ├── Project structure
│   ├── Core endpoints (/engine, /config)
│   ├── WebSocket for progress
│   └── Authentication (if needed)
├── Week 3-4: React project setup
│   ├── Vite + TypeScript configuration
│   ├── Component library setup (MUI or Ant Design)
│   ├── Routing (React Router)
│   └── API client generation (OpenAPI)
└── Week 5-6: Core shared components
    ├── Layout components
    ├── Form components
    ├── Chart wrappers
    └── State management setup

Phase 2: Page Migration (Weeks 7-14)
├── Week 7-8: Engine Design page
├── Week 9-10: Analysis pages (Monte Carlo, Throttle)
├── Week 11-12: Addon pages (Igniter, Injector, Contour)
├── Week 13-14: Project management, Settings
└── Each page: Component development, API integration, testing

Phase 3: Polish (Weeks 15-18)
├── Week 15-16: E2E testing, bug fixes
├── Week 17: Performance optimization
└── Week 18: Documentation, deployment setup
```

### 9.3 Qt Migration Roadmap

```
Phase 1: Foundation (Weeks 1-3)
├── Week 1: Project setup
│   ├── PySide6 project structure
│   ├── Main window and navigation
│   └── Application settings
├── Week 2: Core components
│   ├── Parameter input widgets
│   ├── Plot containers (Plotly or PyQtGraph)
│   └── Progress indicators
└── Week 3: Threading architecture
    ├── Worker thread pattern
    └── Signal/slot communication

Phase 2: Page Development (Weeks 4-12)
├── Weeks 4-5: Engine Design
├── Weeks 6-7: Cooling Analysis
├── Weeks 8-9: Addons (Igniter, Injector, Contour)
├── Weeks 10-11: Analysis (Monte Carlo, Throttle)
└── Week 12: Project management

Phase 3: Distribution (Weeks 13-16)
├── Week 13: PyInstaller/cx_Freeze setup
├── Week 14: Platform-specific builds
├── Week 15: Testing on target platforms
└── Week 16: Documentation, installer creation
```

---

## 10. Conclusion

The frontend migration decision depends primarily on RESA's distribution goals and target users:

1. **For internal/research use with rapid iteration**: Enhanced Streamlit is the pragmatic choice
2. **For commercial web product**: React migration is worth the investment
3. **For desktop engineering tool**: Qt provides the best native experience

The recommended phased approach (enhance Streamlit first, add API layer, then decide) minimizes risk while preserving optionality. The API layer development benefits all paths and should be prioritized regardless of final UI choice.

---

*Document prepared for RESA project frontend architecture decision.*
