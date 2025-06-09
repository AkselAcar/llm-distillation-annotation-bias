FROM nvcr.io/nvidia/pytorch:25.03-py3

# Valeurs par défaut
ARG LDAP_USERNAME=henking
ARG LDAP_UID=224825
ARG LDAP_GROUPNAME=rcp-runai-course-ee-559
ARG LDAP_GID=84650

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} \
 && useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# Copie du code
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Install des paquets système
RUN apt update && apt install -y python3-pip

# Ajout de pip dans le PATH root
ENV PATH="/root/.local/bin:$PATH"

# Install Python deps
WORKDIR /home/${LDAP_USERNAME}
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir accelerate
RUN python3 -c "import accelerate; print('Accelerate version:', accelerate.__version__)"

# Enfin, on change d'utilisateur
USER ${LDAP_USERNAME}
