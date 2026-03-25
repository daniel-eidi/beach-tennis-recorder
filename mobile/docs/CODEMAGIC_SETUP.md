# Codemagic Setup - Beach Tennis Recorder

## Passo a passo para configurar o Codemagic

### 1. Criar conta no Codemagic
- Acesse: https://codemagic.io
- Cadastre-se com sua conta GitHub/GitLab/Bitbucket

### 2. Subir o projeto no GitHub
```bash
cd mobile
git init
git add .
git commit -m "[AGENT-01] feat: initial Flutter project setup"
git remote add origin https://github.com/SEU_USUARIO/beach-tennis-recorder.git
git push -u origin main
```

### 3. Conectar repo no Codemagic
- No dashboard do Codemagic, clique em **"Add application"**
- Selecione o repositorio do GitHub
- O Codemagic vai detectar o `codemagic.yaml` automaticamente

### 4. Configurar iOS Signing (para teste no iPhone)

#### Opcao A: Signing automatico (recomendado)
1. No Codemagic, va em **Settings > Code signing > iOS**
2. Selecione **"Automatic"**
3. Faca login com seu Apple ID
4. O Codemagic gera certificados e provisioning profiles automaticamente

#### Opcao B: Signing manual
1. No Apple Developer Portal (https://developer.apple.com):
   - Crie um **App ID**: `com.beachtennis.recorder`
   - Crie um **Development Certificate** (.p12)
   - Crie um **Development Provisioning Profile**
2. No Codemagic:
   - Va em **Settings > Code signing > iOS**
   - Upload do certificado .p12
   - Upload do provisioning profile
   - Configure o bundle ID: `com.beachtennis.recorder`

### 5. Configurar variaveis de ambiente
No Codemagic, va em **Settings > Environment variables** e crie o grupo `ios_credentials`:
- Nao precisa de variaveis adicionais se usar signing automatico

### 6. Rodar primeiro build
- Faca um push no branch `main` ou `develop`
- Ou va no dashboard e clique **"Start new build"**
- Selecione o workflow **"iOS Development Build"**

### 7. Instalar no iPhone
Apos o build:
- O Codemagic gera um IPA
- Voce pode baixar o IPA direto do dashboard
- Instale via:
  - **AltStore** (sem Mac)
  - **Apple Configurator 2** (com Mac)
  - **TestFlight** (se usar o workflow ios-testflight)

## Workflows disponiveis

| Workflow | Trigger | Saida |
|---|---|---|
| `ios-development` | Push em main/develop | IPA debug |
| `ios-testflight` | Tag `v*` | IPA release via TestFlight |
| `android-development` | Manual | APK debug |

## Para gerar um build TestFlight
```bash
git tag v1.0.0
git push origin v1.0.0
```

## Troubleshooting

### "No valid code signing identity found"
- Verifique se o Apple Developer Account esta configurado no Codemagic
- Para conta gratuita: so funciona Development (nao App Store)

### "Module not found: tflite_flutter"
- O modelo TFLite placeholder e criado automaticamente no build
- O modelo real sera substituido quando o fine-tune terminar

### Build muito lento
- O primeiro build e mais lento (baixa dependencias)
- Builds seguintes usam cache (~5-10 min)
