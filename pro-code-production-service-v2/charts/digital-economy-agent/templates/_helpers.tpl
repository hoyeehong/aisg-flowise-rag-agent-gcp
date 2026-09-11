{{- define "agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agent.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "agent.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "agent.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "agent.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Component-scoped selectors.

The base selector matches every pod in the release, which was fine while the chart had
one workload. With the consumer added it is not: the API Service would select consumer
pods and route HTTP to a process that serves only probes, on a different port. Each
workload therefore carries a component label and selects on it.

A Deployment's selector is immutable, so adding this to an already-installed release
needs a delete and reinstall. It is free now because the chart has never been deployed.
*/}}
{{- define "agent.apiSelectorLabels" -}}
{{ include "agent.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end -}}

{{- define "agent.consumerSelectorLabels" -}}
{{ include "agent.selectorLabels" . }}
app.kubernetes.io/component: consumer
{{- end -}}

{{- define "agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "agent.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Resolve the image reference, preferring a digest over a tag.
A mutable tag means two pods in the same ReplicaSet can run different code after a
rolling restart, which makes an incident unreproducible.
*/}}
{{- define "agent.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}
{{- end -}}
